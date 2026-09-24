// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Shared host-file I/O for storage backends.

use blocking::unblock;
use disk_backend::DiskError;
use pal_async::driver::Driver;
use scsi_buffers::RequestBuffers;
use std::borrow::Borrow;
use std::fs;
use std::io;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use io_uring::opcode;
#[cfg(target_os = "linux")]
use io_uring::types;
#[cfg(target_os = "linux")]
use scsi_buffers::BounceBuffer;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;

/// Owned-buffer and file-management I/O for a shared host file.
#[derive(Clone)]
pub struct OwnedFile {
    file: Arc<fs::File>,
}

impl OwnedFile {
    /// Wraps an open file.
    pub fn new(file: fs::File) -> Self {
        Self {
            file: Arc::new(file),
        }
    }

    fn from_arc(file: Arc<fs::File>) -> Self {
        Self { file }
    }

    /// Reads exactly `buf.len()` bytes at `offset`.
    pub async fn read_into(&self, offset: u64, mut buf: Vec<u8>) -> io::Result<Vec<u8>> {
        let file = self.file.clone();
        unblock(move || {
            read_exact_at(&file, &mut buf, offset)?;
            Ok(buf)
        })
        .await
    }

    /// Writes all of `buf` at `offset`.
    pub async fn write_from(
        &self,
        offset: u64,
        buf: impl Borrow<Vec<u8>> + Send + 'static,
    ) -> io::Result<()> {
        let file = self.file.clone();
        unblock(move || write_all_at(&file, buf.borrow(), offset)).await
    }

    /// Flushes file data and metadata to stable storage.
    pub async fn flush(&self) -> io::Result<()> {
        let file = self.file.clone();
        unblock(move || file.sync_all()).await
    }

    /// Returns the current file size.
    pub fn file_size(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    /// Changes the current file size.
    pub async fn set_file_size(&self, size: u64) -> io::Result<()> {
        let file = self.file.clone();
        unblock(move || file.set_len(size)).await
    }
}

/// Payload I/O scheduler for a shared host file.
pub struct HostFile {
    file: Arc<fs::File>,
    #[cfg(target_os = "linux")]
    driver: Box<dyn Driver>,
}

impl HostFile {
    /// Creates a host-file scheduler using the platform's async I/O engine.
    pub fn new(file: fs::File, driver: impl Driver) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        {
            if !driver.io_uring_probe(opcode::Read::CODE)
                || !driver.io_uring_probe(opcode::Write::CODE)
            {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "driver does not support io_uring file I/O",
                ));
            }

            Ok(Self {
                file: Arc::new(file),
                driver: Box::new(driver),
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            let _ = driver;
            Ok(Self {
                file: Arc::new(file),
            })
        }
    }

    /// Returns an owned-buffer interface for metadata and file management.
    pub fn owned_file(&self) -> OwnedFile {
        OwnedFile::from_arc(self.file.clone())
    }

    /// Reads a positioned file range into request buffers.
    pub async fn read_at(
        &self,
        buffers: &RequestBuffers<'_>,
        file_offset: u64,
    ) -> Result<(), DiskError> {
        self.read_at_impl(buffers, file_offset).await
    }

    /// Writes request buffers to a positioned file range.
    pub async fn write_at(
        &self,
        buffers: &RequestBuffers<'_>,
        file_offset: u64,
        fua: bool,
    ) -> Result<(), DiskError> {
        self.write_at_impl(buffers, file_offset, fua).await
    }

    /// Writes zeroes to a positioned file range.
    pub async fn zero_at(&self, file_offset: u64, len: u64) -> Result<(), DiskError> {
        self.zero_at_impl(file_offset, len).await
    }

    #[cfg(target_os = "linux")]
    async fn read_at_impl(
        &self,
        buffers: &RequestBuffers<'_>,
        file_offset: u64,
    ) -> Result<(), DiskError> {
        let mut bounce = BounceBuffer::new(buffers.len());
        let len = u32::try_from(buffers.len()).map_err(|_| DiskError::InvalidInput)?;

        // SAFETY: `bounce` remains in this async state machine until the
        // submitted operation completes.
        let bytes_read = unsafe {
            self.driver.io_uring_submit(
                opcode::Read::new(
                    types::Fd(self.file.as_raw_fd()),
                    bounce.as_mut_bytes().as_mut_ptr(),
                    len,
                )
                .offset(file_offset)
                .build(),
            )
        }
        .await
        .map_err(DiskError::Io)?;

        check_io_size(bytes_read, buffers.len(), io::ErrorKind::UnexpectedEof)
            .map_err(DiskError::Io)?;
        use guestmem::MemoryWrite;
        buffers.writer().write(bounce.as_mut_bytes())?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    async fn write_at_impl(
        &self,
        buffers: &RequestBuffers<'_>,
        file_offset: u64,
        fua: bool,
    ) -> Result<(), DiskError> {
        use guestmem::MemoryRead;
        let mut bounce = BounceBuffer::new(buffers.len());
        buffers.reader().read(bounce.as_mut_bytes())?;
        let len = u32::try_from(buffers.len()).map_err(|_| DiskError::InvalidInput)?;

        // SAFETY: `bounce` remains in this async state machine until the
        // submitted operation completes.
        let bytes_written = unsafe {
            self.driver.io_uring_submit(
                opcode::Write::new(
                    types::Fd(self.file.as_raw_fd()),
                    bounce.as_mut_bytes().as_ptr(),
                    len,
                )
                .offset(file_offset)
                .rw_flags(if fua { libc::RWF_DSYNC } else { 0 })
                .build(),
            )
        }
        .await
        .map_err(DiskError::Io)?;

        check_io_size(bytes_written, buffers.len(), io::ErrorKind::WriteZero).map_err(DiskError::Io)
    }

    #[cfg(target_os = "linux")]
    async fn zero_at_impl(&self, mut file_offset: u64, mut len: u64) -> Result<(), DiskError> {
        const CHUNK_SIZE: usize = 64 * 1024;
        if len == 0 {
            return Ok(());
        }
        let zeros = BounceBuffer::new(len.min(CHUNK_SIZE as u64) as usize);

        while len != 0 {
            let chunk_len = CHUNK_SIZE.min(len as usize);
            let io_vec = &zeros.io_vecs()[0][..chunk_len];
            // SAFETY: `zeros` remains in this async state machine until every
            // submitted operation that references it completes.
            let bytes_written = unsafe {
                self.driver.io_uring_submit(
                    opcode::Write::new(
                        types::Fd(self.file.as_raw_fd()),
                        io_vec.as_ptr().cast(),
                        chunk_len as u32,
                    )
                    .offset(file_offset)
                    .build(),
                )
            }
            .await
            .map_err(DiskError::Io)?;
            check_io_size(bytes_written, chunk_len, io::ErrorKind::WriteZero)
                .map_err(DiskError::Io)?;
            file_offset += chunk_len as u64;
            len -= chunk_len as u64;
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    async fn read_at_impl(
        &self,
        buffers: &RequestBuffers<'_>,
        file_offset: u64,
    ) -> Result<(), DiskError> {
        self.read_buffered_at(buffers, file_offset).await
    }

    #[cfg(not(target_os = "linux"))]
    async fn read_buffered_at(
        &self,
        buffers: &RequestBuffers<'_>,
        file_offset: u64,
    ) -> Result<(), DiskError> {
        use guestmem::MemoryWrite;
        let buf = self
            .owned_file()
            .read_into(file_offset, vec![0; buffers.len()])
            .await
            .map_err(DiskError::Io)?;
        buffers.writer().write(&buf)?;
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    async fn write_at_impl(
        &self,
        buffers: &RequestBuffers<'_>,
        file_offset: u64,
        fua: bool,
    ) -> Result<(), DiskError> {
        self.write_buffered_at(buffers, file_offset, fua).await
    }

    #[cfg(not(target_os = "linux"))]
    async fn write_buffered_at(
        &self,
        buffers: &RequestBuffers<'_>,
        file_offset: u64,
        fua: bool,
    ) -> Result<(), DiskError> {
        use guestmem::MemoryRead;
        let mut buf = vec![0; buffers.len()];
        buffers.reader().read(&mut buf)?;
        self.owned_file()
            .write_from(file_offset, buf)
            .await
            .map_err(DiskError::Io)?;
        if fua {
            self.owned_file().flush().await.map_err(DiskError::Io)?;
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    async fn zero_at_impl(&self, mut file_offset: u64, mut len: u64) -> Result<(), DiskError> {
        self.zero_buffered_at(file_offset, len).await
    }

    #[cfg(not(target_os = "linux"))]
    async fn zero_buffered_at(&self, mut file_offset: u64, mut len: u64) -> Result<(), DiskError> {
        const CHUNK_SIZE: usize = 64 * 1024;
        while len != 0 {
            let chunk_len = CHUNK_SIZE.min(len as usize);
            self.owned_file()
                .write_from(file_offset, vec![0; chunk_len])
                .await
                .map_err(DiskError::Io)?;
            file_offset += chunk_len as u64;
            len -= chunk_len as u64;
        }
        Ok(())
    }
}

fn check_io_size(actual: i32, expected: usize, kind: io::ErrorKind) -> io::Result<()> {
    if usize::try_from(actual) == Ok(expected) {
        Ok(())
    } else {
        Err(io::Error::new(
            kind,
            format!("host file I/O transferred {actual} of {expected} bytes"),
        ))
    }
}

#[cfg(unix)]
fn file_read_at(file: &fs::File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, offset)
}

#[cfg(windows)]
fn file_read_at(file: &fs::File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buf, offset)
}

#[cfg(unix)]
fn file_write_at(file: &fs::File, buf: &[u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::write_at(file, buf, offset)
}

#[cfg(windows)]
fn file_write_at(file: &fs::File, buf: &[u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_write(file, buf, offset)
}

fn read_exact_at(file: &fs::File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let count = file_read_at(file, buf, offset)?;
        if count == 0 {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        offset += count as u64;
        buf = &mut buf[count..];
    }
    Ok(())
}

fn write_all_at(file: &fs::File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        let count = file_write_at(file, buf, offset)?;
        if count == 0 {
            return Err(io::Error::from(io::ErrorKind::WriteZero));
        }
        offset += count as u64;
        buf = &buf[count..];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use guestmem::GuestMemory;
    use pal_async::DefaultDriver;
    use scsi_buffers::OwnedRequestBuffers;

    #[pal_async::async_test]
    async fn payload_round_trip_and_zero(driver: DefaultDriver) {
        let file = tempfile::tempfile().unwrap();
        file.set_len(256 * 1024).unwrap();
        let file = HostFile::new(file, driver).unwrap();
        let memory = GuestMemory::allocate(128 * 1024);
        let pattern: Vec<u8> = (0..128 * 1024).map(|i| (i % 251) as u8).collect();

        memory.write_at(0, &pattern).unwrap();
        file.write_at(
            &OwnedRequestBuffers::linear(0, 512, false).buffer(&memory),
            0,
            false,
        )
        .await
        .unwrap();
        file.write_at(
            &OwnedRequestBuffers::linear(0, 128 * 1024, false).buffer(&memory),
            64 * 1024,
            true,
        )
        .await
        .unwrap();
        file.zero_at(192 * 1024, 512).await.unwrap();

        memory.fill_at(0, 0, 128 * 1024).unwrap();
        file.read_at(
            &OwnedRequestBuffers::linear(0, 128 * 1024, true).buffer(&memory),
            64 * 1024,
        )
        .await
        .unwrap();
        let mut actual = vec![0; 128 * 1024];
        memory.read_at(0, &mut actual).unwrap();
        assert_eq!(actual, pattern);

        file.read_at(
            &OwnedRequestBuffers::linear(0, 512, true).buffer(&memory),
            192 * 1024,
        )
        .await
        .unwrap();
        memory.read_at(0, &mut actual[..512]).unwrap();
        assert_eq!(actual[..512], [0; 512]);
    }
}
