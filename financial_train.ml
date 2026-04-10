open Matrix
open Nn
open Train
open Financial_loader

let () =
  Random.init 42;
  Printf.printf "--- Financial Data Training (OCaml) ---\n";
  
  let dataset_path = "simulated_financial_forecasting_data.csv" in
  Printf.printf "Loading dataset from %s...\n" dataset_path;
  let (x_data, y_data) = Financial_loader.load_financial_data dataset_path in
  Printf.printf "Loaded %d samples.\n" (List.length x_data);

  (* Architecture: 6 inputs -> 8 hidden -> 2 output classes *)
  let model = Train.init_model [6; 8; 2] in
  
  let epochs = 20 in
  let batch_size = 16 in
  let lr = 0.5 in

  Printf.printf "Starting training (epochs=%d, lr=%.2f)...\n%!" epochs lr;
  let t_start = Unix.gettimeofday () in
  let trained_model = Train.train model x_data y_data epochs batch_size lr in
  let t_end = Unix.gettimeofday () in
  
  let acc = Train.accuracy trained_model x_data y_data in
  Printf.printf "Final Accuracy: %.2f%%\n" (acc *. 100.0);
  Printf.printf "Total Training Time: %.4f seconds\n" (t_end -. t_start);
  Printf.printf "RESULT: %.2f, %.4f, %d\n" (acc *. 100.0) (t_end -. t_start) epochs;
