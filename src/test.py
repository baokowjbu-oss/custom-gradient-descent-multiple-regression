from regressor import BGDRegression
import numpy as np
import time

bgd_regressor =BGDRegression()

def test_regressor_loop(BGDRegressor: BGDRegression):
    print("Running test on partial vectorized batch gradient descent regressor...")
    x_train = np.array([[2.1, 3], [1.4, 2], [0.8, 1], [3.5, 4]])
    y_train = np.array([4.0, 2.5, 1.5, 6.5])
    m, n = x_train.shape
    w_initial = np.zeros((n,))
    b_initial = 0.0
    learning_rate = 0.03
    num_iteration = 10000
    w, b, J_hist, _= BGDRegressor.run_bgd_regressor_loop(x_train, y_train, w_initial, b_initial, learning_rate, num_iteration)

    print("--------------------------------------- The result -----------------------------------------")
    for i in range(n):
        print(f"The weight at minimized J(w,b) w{i} = {w[i]:.4f}")
    print(f"The bias at minimized J(w,b) b = {b:.4f}")
    print(f"The cost function local minimum value J(w,b) = {J_hist[-1]:.4f} \n")

    print("--------------------------------------- Prediction -----------------------------------------")
    for i in range(m):
        prediction = np.dot(x_train[i], w) + b
        print(f"Prediction {i} = {prediction:.4f}. Actual result = {y_train[i]:.4f}")

def test_regressor(BGDRegressor: BGDRegression):
    print("Running test on partial vectorized batch gradient descent regressor...")
    x_train = np.array([[2.1, 3], [1.4, 2], [0.8, 1], [3.5, 4]])
    y_train = np.array([4.0, 2.5, 1.5, 6.5])
    m, n = x_train.shape
    w_initial = np.zeros((n,))
    b_initial = 0.0
    learning_rate = 0.03
    num_iteration = 10000
    w, b, J_hist, _= BGDRegressor.run_bgd_regressor_loop(x_train, y_train, w_initial, b_initial, learning_rate, num_iteration)

    print("--------------------------------------- The result -----------------------------------------")
    for i in range(n):
        print(f"The weight at minimized J(w,b) w{i} = {w[i]:.4f}")
    print(f"The bias at minimized J(w,b) b = {b:.4f}")
    print(f"The cost function local minimum value J(w,b) = {J_hist[-1]:.4f} \n")

    print("--------------------------------------- Prediction -----------------------------------------")
    for i in range(m):
        prediction = np.dot(x_train[i], w) + b
        print(f"Prediction {i} = {prediction:.4f}. Actual result = {y_train[i]:.4f}")

#compare the performance between using loop and using fully vectorization
tic = time.perf_counter()
test_regressor_loop(bgd_regressor)
toc = time.perf_counter()
print("-------------------------------------- Time elasped ----------------------------------------")
print(f"Time take to run partial vectorization batch gradient descent: {1000*(toc - tic):.4f} ms\n")
print("--------------------------------------------------------------------------------------------")

tic = time.perf_counter()
test_regressor(bgd_regressor)
toc = time.perf_counter()
print("-------------------------------------- Time elasped ----------------------------------------")
print(f"Time take to run fully vectorization batch gradient descent: {1000*(toc - tic):.4f} ms\n")
print("--------------------------------------------------------------------------------------------")

