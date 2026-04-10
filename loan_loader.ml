(** Loan Data Loader for OCaml **)

let split_comma s =
  (* Handle potential spaces around commas *)
  String.split_on_char ',' s |> List.map String.trim

let rec read_lines ic acc =
  try
    let line = input_line ic in
    read_lines ic (line :: acc)
  with End_of_file ->
    List.rev acc

(** String mappings for Label Encoding **)
let gender_map s = match s with "female" -> 0.0 | "male" -> 1.0 | _ -> 0.0
let edu_map s = match s with 
  | "Associate" -> 0.0 | "Bachelor" -> 1.0 | "Doctorate" -> 2.0 
  | "High School" -> 3.0 | "Master" -> 4.0 | _ -> 0.0
let home_map s = match s with 
  | "MORTGAGE" -> 0.0 | "OTHER" -> 1.0 | "OWN" -> 2.0 | "RENT" -> 3.0 | _ -> 0.0
let intent_map s = match s with
  | "DEBTCONSOLIDATION" -> 0.0 | "EDUCATION" -> 1.0 | "HOMEIMPROVEMENT" -> 2.0
  | "MEDICAL" -> 3.0 | "PERSONAL" -> 4.0 | "VENTURE" -> 5.0 | _ -> 0.0
let default_map s = match s with "No" -> 0.0 | "Yes" -> 1.0 | _ -> 0.0

let parse_row row =
  match row with
  | [age; gender; edu; income; exp; home; amnt; intent; rate; pct; hist; score; dflt; status] ->
      let features = [
        float_of_string age;
        gender_map gender;
        edu_map edu;
        float_of_string income;
        float_of_string exp;
        home_map home;
        float_of_string amnt;
        intent_map intent;
        float_of_string rate;
        float_of_string pct;
        float_of_string hist;
        float_of_string score;
        default_map dflt
      ] in
      (features, int_of_string status)
  | _ -> failwith "Invalid row format"

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

let load_loan_data filename limit =
  let ic = open_in filename in
  let _header = input_line ic in
  let rec read_count ic count acc =
    if count <= 0 then List.rev acc
    else try
      let line = input_line ic in
      read_count ic (count - 1) (line :: acc)
    with End_of_file -> List.rev acc
  in
  let lines = read_count ic limit [] in
  close_in ic;
  let parsed = List.map (fun l -> parse_row (split_comma l)) lines in
  let x_raw = List.map fst parsed in
  let y = List.map snd parsed in
  let x = normalize_matrix x_raw in
  (x, y)
