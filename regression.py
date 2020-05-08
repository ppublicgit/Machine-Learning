import numpy as np
import matplotlib.pyplot as plt
from Learning.TestInputs import GaussSine
from Learning.mlp import mlp


if __name__ == "__main__":
    forward = mlp(GaussSine().train,
                  GaussSine().trainTarget,
                  valid_inputs=GaussSine().valid,
                  valid_targets=GaussSine().validTarget,
                  validate=True,
                  outtype="linear",
                  nHidden=4,
                  valid_iterations=1000)
    predict = forward(GaussSine().test)
    test_error = np.sum((predict-GaussSine().testTarget)**2)
    print(f"Test error: {test_error}")
    print("")
    plt.figure(1)
    plt.plot(GaussSine().test, GaussSine().testTarget, "r*", label="test")
    plt.plot(GaussSine().test, predict, "g*", label="predict")
    plt.legend()
    plt.show(block=False)
    input("Press enter to exit...")
