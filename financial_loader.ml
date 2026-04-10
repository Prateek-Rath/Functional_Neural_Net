(** Functional Loader for Financial CSV Data **)

let split_comma s =
  String.split_on_char ',' s |> List.map String.trim

let rec read_lines ic acc =
  try
    let line = input_line ic in
    read_lines ic (line :: acc)
  with End_of_file ->
    List.rev acc

let parse_csv filename =
  let ic = open_in filename in
  let _header = input_line ic in
  let lines = read_lines ic [] in
  close_in ic;
  List.map (fun l -> List.map float_of_string (split_comma l)) lines

let normalize_matrix m =
  match m with
  | [] -> []
  | first :: _ ->
      let num_cols = List.length first in
      let rec get_min_max col_idx acc_min acc_max rows =
        match rows with
        | [] -> (acc_min, acc_max)
        | row :: rest ->
            let v = List.nth row col_idx in
            get_min_max col_idx (min acc_min v) (max acc_max v) rest
      in
      let stats = List.init num_cols (fun i -> get_min_max i 1e18 (-1e18) m) in
      List.map (fun row ->
        List.mapi (fun i v ->
          let (min_v, max_v) = List.nth stats i in
          if max_v -. min_v < 1e-9 then 0.0
          else (v -. min_v) /. (max_v -. min_v)
        ) row
      ) m

let load_financial_data filename =
  let data = parse_csv filename in
  (* Features: first 6 columns, Target: last column *)
  let x_raw = List.map (fun row -> 
    let rec take n l = if n <= 0 then [] else List.hd l :: take (n-1) (List.tl l) in
    take 6 row
  ) data in
  let y_raw = List.map (fun row -> List.nth row 6) data in
  
  (* Calculate mean of target *)
  let sum_y = List.fold_left (+.) 0.0 y_raw in
  let mean_y = sum_y /. float_of_int (List.length y_raw) in
  
  (* Binarize y: 0 if < mean, 1 if >= mean *)
  let y = List.map (fun v -> if v < mean_y then 0 else 1) y_raw in
  
  (* Normalize features *)
  let x = normalize_matrix x_raw in
  (x, y)
