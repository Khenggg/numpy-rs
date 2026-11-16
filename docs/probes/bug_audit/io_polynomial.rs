extern crate numpy_lib;
use numpy_lib::*;
use std::io::Cursor;
fn blob(header:&str,payload:&[u8])->Vec<u8> {
    let mut out=b"\x93NUMPY\x02\x00".to_vec();out.extend((header.len()as u32).to_le_bytes());out.extend(header.as_bytes());out.extend(payload);out
}
fn main() {
    assert_eq!(polyadd(&[1.,0.],&[-1.,2.]),vec![0.,2.]);
    assert_eq!(polysub(&[1.,0.],&[1.,2.]),vec![0.,-2.]);
    for (tag,header) in [
        ("badbool","{'descr': '<i8', 'fortran_order': Falsex, 'shape': (1,), }\n"),
        ("badtuple","{'descr': '<i8', 'fortran_order': False, 'shape': (1), }\n"),
        ("dupkey","{'descr': '<i8', 'fortran_order': False, 'shape': (1,), 'shape': (2,), }\n"),
    ] {
        let bytes=blob(header,&7i64.to_le_bytes());
        std::fs::write(format!("target/bug-audit/{tag}.npy"),&bytes).unwrap();
        assert!(read_npy(&mut Cursor::new(bytes)).is_err(),"{tag}");
    }
    println!("PASS B04 B05 B06");
}
