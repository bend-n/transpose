//! Utility for transposing multi-dimensional data stored as a flat slice
//!
//! This library treats Rust slices as flattened row-major 2D arrays, and provides functions to transpose these 2D arrays, so that the row data becomes the column data, and vice versa.
//! ```
//! use std::mem::transmute;
//! // Create a 2D array in row-major order: the rows of our 2D array are contiguous,
//! // and the columns are strided
//! let input_array = vec![ 1, 2, 3,
//! 						4, 5, 6];
//!
//! // Treat our 6-element array as a 2D 3x2 array, and transpose it to a 2x3 array
//! let mut output_array = vec![0; 6];
//! unsafe { mattr::transpose(&input_array, transmute(&mut *output_array), 3, 2) };
//!
//! // The rows have become the columns, and the columns have become the rows
//! let expected_array =  vec![ 1, 4,
//!								2, 5,
//!								3, 6];
//! assert_eq!(output_array, expected_array);
//!
//! // If we transpose our data again, we should get our original data back.
//! let mut final_array = vec![0; 6];
//! unsafe { mattr::transpose(&output_array, transmute(&mut *final_array), 2, 3) };
//! assert_eq!(final_array, input_array);
//! ```
//!
//! This library supports both in-place and out-of-place transposes. The out-of-place
//! transpose is much, much faster than the in-place transpose -- the in-place transpose should
//! only be used in situations where the system doesn't have enough memory to do an out-of-place transpose.
//!
//! The out-of-place transpose uses one out of three different algorithms depending on the length of the input array.
//!
//! - Small: simple iteration over the array.
//! - Medium: divide the array into tiles of fixed size, and process each tile separately.
//! - Large: recursively split the array into smaller parts until each part is of a good size for the tiling algorithm, and then transpose each part.  

#![no_std]
#![allow(internal_features, incomplete_features)]
#![feature(
    const_trait_impl,
    generic_const_exprs,
    maybe_uninit_array_assume_init,
    core_intrinsics,
    const_eval_select
)]

use core::intrinsics::{const_eval_select, transmute_unchecked};
use core::mem::MaybeUninit as MU;
// Block size used by the tiling algoritms
const BLOCK_SIZE: usize = 16;
// Number of segments used by the segmented block transpose function
const NBR_SEGMENTS: usize = 4;
// recursively split data until the number of rows and columns is below this number
const RECURSIVE_LIMIT: usize = 128;

// Largest size for for using the direct approach
const SMALL_LEN: usize = 255;
// Largest size for using the tiled approach
const MEDIUM_LEN: usize = 1024 * 1024;

/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output.
/// Benchmarking shows that loop tiling isn't effective for small arrays.
const unsafe fn transpose_small<T: Copy>(
    input: &[T],
    output: &mut [MU<T>],
    width: usize,
    height: usize,
) {
    let mut x = 0;
    while x != width {
        let mut y = 0;
        while y != height {
            let input_index = x + y * width;
            let output_index = y + x * height;
            (*output.as_mut_ptr().add(output_index)).write(*input.as_ptr().add(input_index));
            y += 1;
        }
        x += 1;
    }
}

// Transpose a subset of the array, from the input into the output. The idea is that by transposing one block at a time, we can be more cache-friendly
// SAFETY: Width * height must equal input.len() and output.len(), start_x + block_width must be <= width, start_y + block height must be <= height
unsafe fn transpose_block<T: Copy>(
    input: &[T],
    output: &mut [MU<T>],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
    block_width: usize,
    block_height: usize,
) {
    for inner_x in 0..block_width {
        for inner_y in 0..block_height {
            let x = start_x + inner_x;
            let y = start_y + inner_y;

            let input_index = x + y * width;
            let output_index = y + x * height;

            (*output.as_mut_ptr().add(output_index)).write(*input.as_ptr().add(input_index));
        }
    }
}

// Transpose a subset of the array, from the input into the output. The idea is that by transposing one block at a time, we can be more cache-friendly
// SAFETY: Width * height must equal input.len() and output.len(), start_x + block_width must be <= width, start_y + block height must be <= height
// This function works as `transpose_block`, but also divides the loop into a number of segments. This makes it more cache fiendly for large sizes.
unsafe fn transpose_block_segmented<T: Copy>(
    input: &[T],
    output: &mut [MU<T>],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
    block_width: usize,
    block_height: usize,
) {
    let height_per_div = block_height / NBR_SEGMENTS;
    for subblock in 0..NBR_SEGMENTS {
        for inner_x in 0..block_width {
            for inner_y in 0..height_per_div {
                let x = start_x + inner_x;
                let y = start_y + inner_y + subblock * height_per_div;

                let input_index = x + y * width;
                let output_index = y + x * height;

                (*output.as_mut_ptr().add(output_index)).write(*input.as_ptr().add(input_index));
            }
        }
    }
}

/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output.
/// This algorithm divides the input into tiles of size BLOCK_SIZE*BLOCK_SIZE,
/// in order to reduce cache misses. This works well for medium sizes, when the
/// data for each tile fits in the caches.  
fn transpose_tiled<T: Copy>(
    input: &[T],
    output: &mut [MU<T>],
    input_width: usize,
    input_height: usize,
) {
    let x_block_count = input_width / BLOCK_SIZE;
    let y_block_count = input_height / BLOCK_SIZE;

    let remainder_x = input_width - x_block_count * BLOCK_SIZE;
    let remainder_y = input_height - y_block_count * BLOCK_SIZE;

    for y_block in 0..y_block_count {
        for x_block in 0..x_block_count {
            unsafe {
                transpose_block(
                    input,
                    output,
                    input_width,
                    input_height,
                    x_block * BLOCK_SIZE,
                    y_block * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                );
            }
        }

        //if the input_width is not cleanly divisible by block_size, there are still a few columns that haven't been transposed
        if remainder_x > 0 {
            unsafe {
                transpose_block(
                    input,
                    output,
                    input_width,
                    input_height,
                    input_width - remainder_x,
                    y_block * BLOCK_SIZE,
                    remainder_x,
                    BLOCK_SIZE,
                );
            }
        }
    }

    //if the input_height is not cleanly divisible by BLOCK_SIZE, there are still a few rows that haven't been transposed
    if remainder_y > 0 {
        for x_block in 0..x_block_count {
            unsafe {
                transpose_block(
                    input,
                    output,
                    input_width,
                    input_height,
                    x_block * BLOCK_SIZE,
                    input_height - remainder_y,
                    BLOCK_SIZE,
                    remainder_y,
                );
            }
        }

        //if the input_width is not cleanly divisible by block_size, there are still a few rows+columns that haven't been transposed
        if remainder_x > 0 {
            unsafe {
                transpose_block(
                    input,
                    output,
                    input_width,
                    input_height,
                    input_width - remainder_x,
                    input_height - remainder_y,
                    remainder_x,
                    remainder_y,
                );
            }
        }
    }
}

/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output.
/// This is a recursive algorithm that divides the array into smaller pieces, until they are small enough to
/// transpose directly without worrying about cache misses.
/// Once they are small enough, they are transposed using a tiling algorithm.
fn transpose_recursive<T: Copy>(
    input: &[T],
    output: &mut [MU<T>],
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    total_columns: usize,
    total_rows: usize,
) {
    let nbr_rows = row_end - row_start;
    let nbr_cols = col_end - col_start;
    if (nbr_rows <= RECURSIVE_LIMIT && nbr_cols <= RECURSIVE_LIMIT)
        || nbr_rows <= 2
        || nbr_cols <= 2
    {
        let x_block_count = nbr_cols / BLOCK_SIZE;
        let y_block_count = nbr_rows / BLOCK_SIZE;

        let remainder_x = nbr_cols - x_block_count * BLOCK_SIZE;
        let remainder_y = nbr_rows - y_block_count * BLOCK_SIZE;

        for y_block in 0..y_block_count {
            for x_block in 0..x_block_count {
                unsafe {
                    transpose_block_segmented(
                        input,
                        output,
                        total_columns,
                        total_rows,
                        col_start + x_block * BLOCK_SIZE,
                        row_start + y_block * BLOCK_SIZE,
                        BLOCK_SIZE,
                        BLOCK_SIZE,
                    );
                }
            }

            //if the input_width is not cleanly divisible by block_size, there are still a few columns that haven't been transposed
            if remainder_x > 0 {
                unsafe {
                    transpose_block(
                        input,
                        output,
                        total_columns,
                        total_rows,
                        col_start + x_block_count * BLOCK_SIZE,
                        row_start + y_block * BLOCK_SIZE,
                        remainder_x,
                        BLOCK_SIZE,
                    );
                }
            }
        }

        //if the input_height is not cleanly divisible by BLOCK_SIZE, there are still a few rows that haven't been transposed
        if remainder_y > 0 {
            for x_block in 0..x_block_count {
                unsafe {
                    transpose_block(
                        input,
                        output,
                        total_columns,
                        total_rows,
                        col_start + x_block * BLOCK_SIZE,
                        row_start + y_block_count * BLOCK_SIZE,
                        BLOCK_SIZE,
                        remainder_y,
                    );
                }
            }

            //if the input_width is not cleanly divisible by block_size, there are still a few rows+columns that haven't been transposed
            if remainder_x > 0 {
                unsafe {
                    transpose_block(
                        input,
                        output,
                        total_columns,
                        total_rows,
                        col_start + x_block_count * BLOCK_SIZE,
                        row_start + y_block_count * BLOCK_SIZE,
                        remainder_x,
                        remainder_y,
                    );
                }
            }
        }
    } else if nbr_rows >= nbr_cols {
        transpose_recursive(
            input,
            output,
            row_start,
            row_start + (nbr_rows / 2),
            col_start,
            col_end,
            total_columns,
            total_rows,
        );
        transpose_recursive(
            input,
            output,
            row_start + (nbr_rows / 2),
            row_end,
            col_start,
            col_end,
            total_columns,
            total_rows,
        );
    } else {
        transpose_recursive(
            input,
            output,
            row_start,
            row_end,
            col_start,
            col_start + (nbr_cols / 2),
            total_columns,
            total_rows,
        );
        transpose_recursive(
            input,
            output,
            row_start,
            row_end,
            col_start + (nbr_cols / 2),
            col_end,
            total_columns,
            total_rows,
        );
    }
}

/// Transpose the input array into the output array.
///
/// Given an input array of size input_width * input_height, representing flattened 2D data stored in row-major order,
/// transpose the rows and columns of that input array into the output array
/// # Safety
///
/// UB if `input.len() != input_width * input_height` or if `output.len() != input_width * input_height`
pub const unsafe fn transpose<T: Copy>(
    input: &[T],
    output: &mut [MU<T>],
    input_width: usize,
    input_height: usize,
) {
    const fn const_version<T: Copy>(a: &[T], b: &mut [MU<T>], c: usize, d: usize) {
        unsafe { transpose_small(a, b, c, d) }
    }
    fn normal<T: Copy>(a: &[T], b: &mut [MU<T>], c: usize, d: usize) {
        unsafe { transpose_(a, b, c, d) }
    }
    const_eval_select(
        (input, output, input_width, input_height),
        const_version,
        normal,
    )
}
/// Transposed array. Good!
/// ```
/// #![feature(generic_const_exprs)]
/// let input_array = [1, 2, 3,
///                    4, 5, 6];
/// assert_eq!(mattr::transposed::<u8, 3, 2>(input_array),
///     [ 1, 4,
///       2, 5,
///       3, 6]
/// );
/// ```
pub const fn transposed<T: Copy, const W: usize, const H: usize>(input: [T; W * H]) -> [T; W * H] {
    let mut output = [const { MU::uninit() }; W * H];
    unsafe { transpose(&input, &mut output, W, H) };
    unsafe { MU::array_assume_init(output) }
}

/// Transpose array, but like, not flat.
///
/// ```
/// #![feature(generic_const_exprs)]
/// let input_array = [[1, 2, 3],
///                    [4, 5, 6]];
/// assert_eq!(mattr::transpose_array(input_array), [[1, 4], [2, 5], [3, 6]]);
/// ```
pub const fn transpose_array<T: Copy, const W: usize, const H: usize>(
    input: [[T; W]; H],
) -> [[T; H]; W]
where
    [(); W * H]:,
{
    unsafe {
        let t = transposed::<T, W, H>(transmute_unchecked(input));
        t.as_ptr().cast::<[[T; H]; W]>().read()
    }
}

unsafe fn transpose_<T: Copy>(
    input: &[T],
    output: &mut [MU<T>],
    input_width: usize,
    input_height: usize,
) {
    if input_width * input_height != output.len() || input_width * input_height != input.len() {
        #[cfg(debug_assertions)]
        panic!();
        #[cfg(not(debug_assertions))]
        unsafe {
            core::hint::unreachable_unchecked()
        };
    }
    if input.len() <= SMALL_LEN {
        unsafe { transpose_small(input, output, input_width, input_height) };
    } else if input.len() <= MEDIUM_LEN {
        transpose_tiled(input, output, input_width, input_height);
    } else {
        transpose_recursive(
            input,
            output,
            0,
            input_height,
            0,
            input_width,
            input_width,
            input_height,
        );
    }
}
