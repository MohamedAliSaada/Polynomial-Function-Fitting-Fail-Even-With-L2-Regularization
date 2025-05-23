# Polynomial Function Fitting  Fail Even With L2 Regularization

This project demonstrates the challenge of fitting even a **simple polynomial function** using a neural network with **L2 regularization**, and highlights the limitations when using only basic dense layers.

---

## 📁 Files Included

* `regularization.py`: Main training script using L2 regularization
* `Module_summary.png`: Network architecture summary
* `fit_resuts.png`: MAE and MSE over training epochs
* `predict_results_to_test_generalization.png`: Comparison of prediction vs true polynomial curve

---

## 📉 Target Function

We try to approximate the following function:

$$
    y = x^3 + 3x + 5x^9
$$

with $x \in [-100, 100]$

Data is normalized using `StandardScaler`.

---

## 📊 Model Architecture

* Dense(128, relu) + L2
* Dense(64, relu) + L2
* Dense(32, relu) + L2
* Dense(8, relu) + L2
* Dense(1, linear)

Regularization strength: $\lambda = 0.1$

> Total parameters: 32,597
> Trainable: 10,865

---

## 🏋️ Training Results

* Training loss decreased as expected
* Validation loss plateaued early, showing no clear generalization

---

## 🔍 Prediction vs Ground Truth

Despite regularization, the network fails to generalize to the function well:

* Predicted output is **mostly flat** across the range
* Model fails to capture even the basic non-linearity of the true function

---

## ❌ Lesson Learned

* Even with **L2 regularization**, a model may fail to capture complex nonlinear functions
* Function shape and network expressivity matter more than size alone
* Consider other methods like:

  * Polynomial regression
  * Fourier features
  * Custom activations (e.g. sine)

---

## 🔹 Next Steps

* Try deeper architecture or different activations (e.g. `tanh`, `swish`)
* Use `Dropout` in conjunction with `L2`
* Try `Fourier Features` or positional encoding

---

## ✍️ Author

Documented by \[سعاده].
