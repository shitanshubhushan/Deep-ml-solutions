## Question Link
https://www.deep-ml.com/problems/6

## My Solution
```
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    eigenvalues = []
    a,b = matrix[0]
    c,d = matrix[1]

    trace = a+d
    det = a*d-b*c

    mid_part = trace**2 - 4*det
    if mid_part < 0:
        real = trace/2
        img = (abs(mid_part)**0.5)/2
        eigenvalues.append(complex(real,img))
        eigenvalues.append(complex(real,-img))
    else:
        eigenvalues.append((trace+(mid_part**0.5))/2)
        eigenvalues.append((trace-(mid_part**0.5))/2)

    eigenvalues.sort(reverse=True)
    return eigenvalues
```

* The above only works for 2x2 matrix
* det(a-lambda*i)= 0
* lambda^2 - trace(a)*lambda + det(a) = 0, solve for lambda
* python sort() uses reverse