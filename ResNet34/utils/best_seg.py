res = []
with open('../result2.txt', 'r', encoding='utf-8') as files:
    for file in files:
        file=file.replace('\n','')
        res.append(file.split(' '))
for i in range(9):
    for j in range(9):
        res[i][j] = int(res[i][j])
s = []
sums = 0
aa = [-1]*2
mm = -1
for i in range(9):
    for j in range(9):
        sums+=res[i][j]
for i in range(1,7):
    aa[0]=i
    for j in range(i+1,8):
        aa[1]=j
        m = j+1
        sum = 0
        for ii in range(9):
            s0,s1=-1,-1
            if 0<=ii<aa[0]:
                s0=0
                s1=aa[0]
            elif aa[0]<=ii<aa[1]:
                s0=aa[0]
                s1=aa[1]
            else:
                s0=aa[1]
                s1=9
            for jj in range(s0,s1):
                sum+=res[ii][jj]

        s.append([sum/sums,aa[0],aa[1]])
s = sorted(s,reverse=False)
# print(s[:5])
concentrations = [0,50,60,100,200,300,400,500,1000]
for i in s:
    print(f"分割浓度：{concentrations[i[1]]},{concentrations[i[2]]},准确率为:{i[0]:.3f}")

