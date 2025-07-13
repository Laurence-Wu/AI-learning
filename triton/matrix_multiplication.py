
import torch


def matrix_multiplication_torch(x,y):
    x= torch.tensor(x)
    y = torch.tensor(y)
    output = torch.zeros(x.size(0),y.size(1))
    for x_row_index in range(x.size(0)):
        for y_col_index in range(y.size(1)):
            output[x_row_index][y_col_index] = torch.dot(
            x[x_row_index],
            y.transpose(1,0)[y_col_index]
            )
    return output
a = [[0,1],[1,0]]
b = [[12,12,98],[23,24,53]]

print(matrix_multiplication_torch(a,b))
