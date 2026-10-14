#![cfg(feature = "http")]
use std::{
    io::{Read, Write},
    net::TcpListener,
};
use tablebase_core::archive::RangeStore;
use tablebase_native::transport::HttpRanges;
#[test]
fn exact_http_resource_and_bad_responses() {
    let key = "0".repeat(64) + ".tbobj";
    for (status, length, body, ok) in [
        (200, 8, vec![1; 8], true),
        (200, 7, vec![1; 7], false),
        (404, 8, vec![1; 8], false),
        (206, 8, vec![1; 8], false),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let k = key.clone();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(std::time::Duration::from_secs(3)))
                .unwrap();
            let mut request = Vec::new();
            let mut byte = [0; 1];
            while !request.ends_with(b"\r\n\r\n") {
                socket.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            assert!(
                String::from_utf8_lossy(&request)
                    .starts_with(&format!("GET /range/{k}/7/8 HTTP/1.1"))
            );
            socket.write_all(format!("HTTP/1.1 {status} test\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n").as_bytes()).unwrap();
            socket.write_all(&body).unwrap();
        });
        let store = HttpRanges::new(&format!("http://{address}")).unwrap();
        assert_eq!(store.read(&key, 7, 8).is_ok(), ok);
        server.join().unwrap();
        assert!(store.read(&key, u64::MAX, 8).is_err());
        assert!(store.read("bad", 0, 8).is_err());
        assert!(store.read(&key, 0, 0).is_err());
    }
    for base in [
        "ftp://example.com",
        "https://u:p@example.com",
        "https://example.com?x=1",
        "https://example.com/#x",
    ] {
        assert!(HttpRanges::new(base).is_err());
    }
}
