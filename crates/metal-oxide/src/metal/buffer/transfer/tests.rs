use super::{pattern, vector_size};

#[test]
fn patterns_repeat_an_element_to_whole_vectors() {
    assert_eq!(pattern(7_u8), [7; 16]);
    assert_eq!(
        pattern(0x0403_0201_u32),
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    );
    let triple = pattern([1_u8, 2, 3]);
    assert_eq!(triple.len(), 48);
    assert!(triple.chunks(3).all(|chunk| chunk == [1, 2, 3]));
    let record = pattern((1_u32, 2_u32, 3_u32));
    assert_eq!(record.len(), 48);
    assert_eq!(&record[..12], [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
}

#[test]
fn uniform_elements_fill_with_one_vector() {
    assert_eq!(pattern((0_u32, 0_u32, 0_u32)), [0; 16]);
    assert_eq!(pattern([9_u8; 5]), [9; 16]);
}

#[test]
fn sizes_count_whole_vectors_and_tail_bytes() {
    assert_eq!(vector_size(0).unwrap(), [0, 0, 1, 0]);
    assert_eq!(vector_size(35).unwrap(), [2, 3, 1, 0]);
    let largest = (u32::MAX as usize - 4) * 16 + 15;
    assert_eq!(vector_size(largest).unwrap()[..2], [u32::MAX - 4, 15]);
    assert!(vector_size(largest + 1).is_err());
}
