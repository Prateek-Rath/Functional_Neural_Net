(** Pure functional matrix library using float list list **)

type matrix = float list list

(** Transpose a matrix **)
let rec transpose = function
  | [] | [] :: _ -> []
  | rows ->
      List.map List.hd rows :: transpose (List.map List.tl rows)

(** Vector dot product **)
let dot_product v1 v2 =
  List.fold_left2 (fun acc x y -> acc +. x *. y) 0.0 v1 v2

(** Matrix multiplication: A x B **)
let matmul a b =
  let b_t = transpose b in
  List.map (fun row -> List.map (fun col -> dot_product row col) b_t) a

(** Element-wise mapping **)
let map f m = List.map (List.map f) m

(** Element-wise mapping with two matrices **)
let map2 f m1 m2 =
  List.map2 (List.map2 f) m1 m2

(** Matrix addition with broadcasting for bias (1xN added to MxN) **)
let add_bias matrix = function
  | [bias_row] -> List.map (fun row -> List.map2 (+.) row bias_row) matrix
  | _ -> failwith "Invalid bias dimensions for broadcasting"

(** Matrix addition (standard) **)
let add = map2 (+.)

(** Matrix subtraction **)
let sub = map2 (-.)

(** Scalar multiplication **)
let mul_scalar s = map (fun x -> x *. s)

(** Sum along axis 0 (columns) **)
let sum_axis_0 m =
  match m with
  | [] -> []
  | first :: _ ->
      let cols = List.length first in
      let init = List.init cols (fun _ -> 0.0) in
      [List.fold_left (fun acc row -> List.map2 (+.) acc row) init m]

(** Element-wise product **)
let mul_elementwise = map2 ( *. )

(** Helper to create a matrix with a single value **)
let make rows cols v =
  List.init rows (fun _ -> List.init cols (fun _ -> v))

(** Flatten a matrix (batch) into a single list of floats (if needed) **)
let flatten m = List.concat m
