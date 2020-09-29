import sys
try:    #if numpy is not found
    import numpy as np
except Exception as e:
    print(e)
    print("Install numpy with 'pip3 install numpy' command (or pip)"); exit()

try:    #if PIL is not found
    from PIL import Image
except Exception as e:
    print(e)
    print("Install PIL with 'pip3 install pillow' command (or pip)"); exit()

# np.set_printoptions(threshold=np.inf) #enable this while debugging

def flat(x):
    return 1 if x<129 else 0

arr = []

for i in range(10):
    with Image.open(f"train_data/{i}.jpg") as img:
        # img.show()
        a = np.array((np.asarray(img.convert('L')))<129, dtype=int) #making the image b&w
        print(i,'>',a.shape)
        arr.append(a)   #appending to the list

#printing
# for x in arr:
#     np.savetxt(sys.stdout, x, fmt='%i')
#     print()

arr = np.array(arr)
np.save('processed_data.npy', arr)    #creating the final pocessed data