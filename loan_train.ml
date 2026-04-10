open Matrix
open Nn
open Train
open Loan_loader

let () =
  Random.init 42;
  Printf.printf "--- Loan Data Training (OCaml) ---\n";
  
  let dataset_path = "loan_data.csv" in
  let limit = 10000 in
  Printf.printf "Loading %d samples from %s...\n" limit dataset_path;
  let (x_data, y_data) = Loan_loader.load_loan_data dataset_path limit in
  Printf.printf "Loaded %d samples.\n" (List.length x_data);

  (* Architecture: 13 inputs -> 16 hidden -> 2 output classes *)
  let model = Train.init_model [13; 16; 2] in
  
  let epochs = 20 in
  let batch_size = 32 in
  let lr = 0.1 in

  Printf.printf "Starting training (epochs=%d, lr=%.2f)...\n%!" epochs lr;
  let t_start = Unix.gettimeofday () in
  let trained_model = Train.train model x_data y_data epochs batch_size lr in
  let t_end = Unix.gettimeofday () in
  
  let acc = Train.accuracy trained_model x_data y_data in
  Printf.printf "Final Accuracy: %.2f%%\n" (acc *. 100.0);
  Printf.printf "Total Training Time: %.4f seconds\n" (t_end -. t_start);
  Printf.printf "RESULT: %.2f, %.4f, %d\n" (acc *. 100.0) (t_end -. t_start) epochs;
