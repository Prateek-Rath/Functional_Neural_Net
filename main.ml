open Matrix
open Nn
open Train

let () =
  Random.self_init ();
  Printf.printf "--- Functional Neural Network Demo ---\n";
  
  (* 1. XOR Dataset Demonstration *)
  Printf.printf "\n[1] Training on XOR dataset...\n";
  let x_xor = [
    [0.0; 0.0];
    [0.0; 1.0];
    [1.0; 0.0];
    [1.0; 1.0]
  ] in
  let y_xor = [0; 1; 1; 0] in

  (* Generic model structure: 2 input -> 8 hidden -> 2 output classes *)
  let model_xor = Train.init_model [2; 8; 2] in
  let trained_xor = Train.train model_xor x_xor y_xor 1000 4 0.5 in
  
  let xor_acc = Train.accuracy trained_xor x_xor y_xor in
  Printf.printf "XOR Accuracy: %.2f%%\n" (xor_acc *. 100.0);

  (* 2. MNIST Setup (Instructions) *)
  Printf.printf "\n[2] MNIST Information:\n";
  Printf.printf "To run with MNIST images, ensure you have the following files in a './data/' folder:\n";
  Printf.printf "- train-images-idx3-ubyte\n";
  Printf.printf "- train-labels-idx1-ubyte\n";
  Printf.printf "\nThen use: let x = Mnist.load_mnist_images \"data/train-images-idx3-ubyte\" in ...\n";
  Printf.printf "Note: The functional implementation is generic and ready for MNIST!\n";
