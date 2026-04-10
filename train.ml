open Nn
open Matrix

(** Initialize random weights **)
let init_layer input_size output_size =
  let scale = 0.01 in
  let w = List.init input_size (fun _ -> 
    List.init output_size (fun _ -> (Random.float 2.0 -. 1.0) *. scale)
  ) in
  let b = Matrix.make 1 output_size 0.0 in
  { w; b }

let rec init_model layers_sizes =
  match layers_sizes with
  | [] | [_] -> []
  | in_size :: out_size :: rest ->
      init_layer in_size out_size :: init_model (out_size :: rest)

(** Functional Batching **)
let rec get_batch data start size acc =
  if size <= 0 then (List.rev acc, data)
  else match data with
    | [] -> (List.rev acc, [])
    | x :: xs -> get_batch xs (start + 1) (size - 1) (x :: acc)

(** Training step for one batch **)
let train_batch model x_batch y_batch lr =
  let (acts, zs) = Nn.forward model x_batch in
  let (grads_w, grads_b) = Nn.backward model acts zs y_batch in
  Nn.update model grads_w grads_b lr

(** One Epoch **)
let rec train_epoch model x_data y_data batch_size lr current_idx total =
  if current_idx >= total then model
  else
    let (x_batch, _) = get_batch (List.filteri (fun i _ -> i >= current_idx) x_data) 0 batch_size [] in
    let (y_batch, _) = get_batch (List.filteri (fun i _ -> i >= current_idx) y_data) 0 batch_size [] in
    let new_model = train_batch model x_batch y_batch lr in
    train_epoch new_model x_data y_data batch_size lr (current_idx + batch_size) total

(** Full Training Loop **)
let rec train model x_data y_data epochs batch_size lr =
  if epochs <= 0 then model
  else
    let new_model = train_epoch model x_data y_data batch_size lr 0 (List.length x_data) in
    let acts, _ = Nn.forward new_model x_data in
    let loss = Nn.compute_loss (List.hd acts) y_data in
    Printf.printf "Epoch remaining: %d, Loss: %f\n%!" epochs loss;
    train new_model x_data y_data (epochs - 1) batch_size lr

(** Prediction / Accuracy **)
let predict_from_probs probs =
  List.map (fun row ->
    let max_idx = ref 0 in
    let max_val = ref (List.hd row) in
    List.iteri (fun i v ->
      if v > !max_val then (max_val := v; max_idx := i)
    ) row;
    !max_idx
  ) probs

let accuracy model x y =
  let acts, _ = Nn.forward model x in
  let preds = predict_from_probs (List.hd acts) in
  let correct = List.fold_left2 (fun acc p t -> if p = t then acc + 1 else acc) 0 preds y in
  float_of_int correct /. float_of_int (List.length y)
