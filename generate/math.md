\text{Problem 1: Evaluate} \int_{0}^{\frac{\pi}{2}} e^{2x} \cos(x) dx
\text{Problem 2: Evaluate} \int_{0}^{\frac{\pi}{2}} \sin^8(x) dx
\text{Problem 3: Evaluate} \int_{-\infty}^{\infty} e^{-ax^2} dx, \quad a > 0
\text{Problem 4: Evaluate} \int_{0}^{\infty} \frac{\sin(x)}{x} dx
\text{Problem 5: Evaluate} \int_{0}^{1} \frac{x^3 - 1}{\ln(x)} dx
\text{Problem 6: Evaluate} \int_{0}^{\infty} x^4 e^{-x} dx
\text{Problem 7: Evaluate} \int_{2\sqrt{2}}^{4} \frac{1}{x \sqrt{x^2 - 4}} dx
\text{Problem 8: Evaluate} \int_{0}^{2} \frac{1}{(x-1)^{2/3}} dx
\text{Problem 9: Evaluate} \int_{0}^{\frac{\pi}{2}} \ln(\tan(x)) dx
\text{Problem 10: Evaluate} \int_{0}^{1} x^3 (1-x)^4 dx


code:
# Question: What is the role of the 'pivot' in Quicksort?
def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1