   
#My Username and Password are stored in a .txt file stored in a data folder which belong to the gitignore
with open('data/copernicus_login.txt') as f:
    lines = f.readlines()
        
USERNAME = lines[0][1:-1]
PASSWORD = lines[1][:-1]

print(len(PASSWORD))
print(PASSWORD)
