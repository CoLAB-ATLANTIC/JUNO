

with open('JUNO/data/copernicus_login.txt') as f:   #quando fizer clone para o servidor esta documento .txt vai ser ignorado
    lines = f.readlines()
        
USERNAME = lines[0][1:-1]
PASSWORD = lines[1][:-1]

print(len(PASSWORD))
print(PASSWORD)