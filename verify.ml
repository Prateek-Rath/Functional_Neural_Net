open Matrix
open Nn
open Train

let () =
  Random.init 42;
  
  (* 2 inputs, 3 hidden, 2 output classes *)  
  (* XOR data *)
  let x = [
    [0.0; 0.0];
    [0.0; 1.0];
    [1.0; 0.0];
    [1.0; 1.0]
  ] in
  let y = [0; 1; 1; 0] in

  (* Fixed weights for deterministic comparison: 2 -> 4 -> 2 *)
  let w1 = [
    [0.1; 0.2; 0.3; 0.4];
    [0.5; 0.6; 0.7; 0.8]
  ] in
  let b1 = [[0.01; 0.02; 0.03; 0.04]] in
  let w2 = [
    [0.1; 0.2];
    [0.3; 0.4];
    [0.5; 0.6];
    [0.7; 0.8]
  ] in
  let b2 = [[0.05; 0.06]] in

  let model = [
    { w = w1; b = b1 };
    { w = w2; b = b2 }
  ] in

  (* One step of training *)
  let lr = 0.5 in
  let (acts, zs) = Nn.forward model x in
  let loss = Nn.compute_loss (List.hd acts) y in
  Printf.printf "Initial loss: %f\n" loss;

  let (grads_w, grads_b) = Nn.backward model acts zs y in
  let updated_model = Nn.update model grads_w grads_b lr in

  let (acts2, _) = Nn.forward updated_model x in
  let loss2 = Nn.compute_loss (List.hd acts2) y in
  Printf.printf "Loss after 1 step: %f\n" loss2;
