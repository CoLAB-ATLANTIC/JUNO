from datetime import date

start = []
end = []

#Create 2 list with the start and end of summer dates for the last 10 years (2011 to 2021)
for i in range(1, 10+2):        #range começa em 1 para ignorarmos 2022 e vai até years+2 (neste caso 2011)
    start.append(str((date.today().year-i)) + '0601')    # from_start_date='0601' -> começa a 1 de junho
    end.append(str((date.today().year-i)) + '0831')          # 'to_end_date = 0831' -> termina a 31 de Agosto
    
for j in range(0, len(start)):    
    print(start[j][:4])
    #print(end)