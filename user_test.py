import torch
shifts_x = torch.arange(
        0, 3, step=1,
        dtype=torch.float32
    )
shifts_y = torch.arange(
        0, 4, step=1,
        dtype=torch.float32
    )
# print("shifts_x：", shifts_x)
# print("shifts_y：", shifts_y)
# shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
# print("shift_x：", shifts_x)
# print("shift_y：", shifts_y)
# shift_x = shift_x.reshape(-1)
# shift_y = shift_y.reshape(-1)
# print("shift_x：", shifts_x)
# print("shift_y：", shifts_y)
# locations = torch.stack((shift_x, shift_y), dim=1)
#
# print(locations)
# locations = locations[:, None, :].expand(-1, 4, -1)
# print(locations)

list = [[1,2,3],[4,5,6]]
print(list[0].index(3))
