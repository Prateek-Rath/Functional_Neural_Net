(** Functional MNIST Loader **)

let read_int32_bin ic =
  let b1 = input_byte ic in
  let b2 = input_byte ic in
  let b3 = input_byte ic in
  let b4 = input_byte ic in
  (b1 lsl 24) lor (b2 lsl 16) lor (b3 lsl 8) lor b4

let rec read_bytes ic n acc =
  if n <= 0 then List.rev acc
  else read_bytes ic (n - 1) (float_of_int (input_byte ic) :: acc)

let rec read_images ic n img_size acc =
  if n <= 0 then List.rev acc
  else
    let img = read_bytes ic img_size [] in
    let normalized_img = List.map (fun x -> x /. 255.0) img in
    read_images ic (n - 1) img_size (normalized_img :: acc)

let load_mnist_images filename =
  let ic = open_in_bin filename in
  let _magic = read_int32_bin ic in
  let count = read_int32_bin ic in
  let rows = read_int32_bin ic in
  let cols = read_int32_bin ic in
  let images = read_images ic count (rows * cols) [] in
  close_in ic;
  images

let rec read_labels ic n acc =
  if n <= 0 then List.rev acc
  else read_labels ic (n - 1) (input_byte ic :: acc)

let load_mnist_labels filename =
  let ic = open_in_bin filename in
  let _magic = read_int32_bin ic in
  let count = read_int32_bin ic in
  let labels = read_labels ic count [] in
  close_in ic;
  labels

(** Functional helper to get a subset of data **)
let take n l =
  let rec loop i acc l =
    if i <= 0 then List.rev acc
    else match l with
      | [] -> List.rev acc
      | x :: xs -> loop (i - 1) (x :: acc) xs
  in
  loop n [] l
