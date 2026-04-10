open Matrix

type layer = {
  w : matrix;
  b : matrix; (* 1 x N matrix *)
}

type model = layer list

(** Activation: ReLU **)
let relu_f x = if x > 0. then x else 0.
let relu m = Matrix.map relu_f m

let relu_deriv_f x = if x > 0. then 1. else 0.
let relu_derivative m = Matrix.map relu_deriv_f m

(** Activation: Softmax (row-wise) **)
let softmax m =
  List.map (fun row ->
    let max_val = List.fold_left max (-1e9) row in
    let exps = List.map (fun x -> exp (x -. max_val)) row in
    let sum_exps = List.fold_left (+.) 0.0 exps in
    List.map (fun x -> x /. (sum_exps +. 1e-9)) exps
  ) m

(** Forward Pass **)
(* returns (activations, z_values) in reverse order for backprop *)
let forward model input =
  let rec loop layers act_acc z_acc =
    match layers with
    | [] -> (act_acc, z_acc)
    | layer :: rest ->
        let last_act = List.hd act_acc in
        let z = Matrix.add_bias (Matrix.matmul last_act layer.w) layer.b in
        let next_act =
          if rest = [] then softmax z else relu z
        in
        loop rest (next_act :: act_acc) (z :: z_acc)
  in
  loop model [input] []

(** Loss: Cross Entropy **)
(* y_true is a list of integers (class indices) *)
let compute_loss y_pred y_true =
  let m = float_of_int (List.length y_true) in
  let log_likelihoods = List.map2 (fun pred_row true_idx ->
    let p = List.nth pred_row true_idx in
    -. (log (p +. 1e-9))
  ) y_pred y_true in
  (List.fold_left (+.) 0.0 log_likelihoods) /. m

(** Backward Pass **)
let backward model activations z_values y_true =
  let m = float_of_int (List.length y_true) in
  
  (* Initial dz for output layer (softmax + cross entropy) *)
  let last_act = List.hd activations in
  let dz_init = List.mapi (fun i row ->
    let true_idx = List.nth y_true i in
    List.mapi (fun j x ->
      if j = true_idx then (x -. 1.0) /. m else x /. m
    ) row
  ) last_act in

  let rec loop layers acts zs dz grads_w grads_b =
    match layers, acts with
    | layer :: layer_rest, a_prev :: a_rest ->
        (* Current layer gradients *)
        let dw = Matrix.matmul (Matrix.transpose a_prev) dz in
        let db = Matrix.sum_axis_0 dz in
        
        let grads_w = dw :: grads_w in
        let grads_b = db :: grads_b in

        if layer_rest = [] then
          (grads_w, grads_b)
        else
          let da = Matrix.matmul dz (Matrix.transpose layer.w) in
          match zs with
          | z_prev :: z_next_rest ->
              let dz_next = Matrix.mul_elementwise da (relu_derivative z_prev) in
              loop layer_rest a_rest z_next_rest dz_next grads_w grads_b
          | _ -> failwith "Insufficient z_values for backprop"
    | _ -> (grads_w, grads_b)
  in
  (* model is in forward order, activations are [a_last, ..., a_0], z_values are [z_last, ..., z_1] *)
  loop (List.rev model) (List.tl activations) (List.tl z_values) dz_init [] []

(** Update Weights **)
let map3 f l1 l2 l3 =
  List.map2 (fun (x1, x2) x3 -> f x1 x2 x3) (List.combine l1 l2) l3

let update model grads_w grads_b lr =
  map3 (fun layer dw db ->
    {
      w = Matrix.sub layer.w (Matrix.mul_scalar lr dw);
      b = Matrix.sub layer.b (Matrix.mul_scalar lr db);
    }
  ) model grads_w grads_b
