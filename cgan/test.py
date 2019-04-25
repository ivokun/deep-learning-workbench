arrays = [0,1,0,1]
state =0
ans = 0
for i in range(arrays):
  if array[i] == state:
    ans += 1
    state = 1 - state

print(ans)