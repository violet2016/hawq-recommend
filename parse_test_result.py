import sys
import re
import psycopg2
filename = sys.argv[1]
print("filename", filename)
query_id = sys.argv[2]
test_id = sys.argv[3]
hostname = 'localhost'
username = 'vcheng'
database = 'hawq-recommend'
myConnection = psycopg2.connect( host=hostname, user=username, dbname=database )

with myConnection.cursor() as cur:
    with open(filename, "r") as f:
        content = f.readlines()
        count_index = 0
        cpu = 0.0
        seg = 0.0
        mem = 0.0
        score = 0.0
        for c in content:
            stripped_c = c.lstrip()

            m_env = re.search(r'^env: \[\[\[\s*\d+.\s+(\d+).\s(\d+).\s*\]\]\]', stripped_c)
            if m_env is not None:
                seg =m_env.group(1)
                mem =  m_env.group(2)
                
            #cpu: [[[2000.]]]
            #pred: -0.08866341
            m_cpu = re.search(r'^cpu: \[\[\[\s*(\d+).\s*\]\]\]', stripped_c)   
            if m_cpu is not None:
                cpu = m_cpu.group(1)
            m_pred = re.search(r'^pred:\s*([0-9\-.]+)', stripped_c)   
            if m_pred is not None:
                score = m_pred.group(1)
            count_index = count_index + 1
            if count_index == 3:
                #print("new one")
                print("index %s seg %s mem %s cpu %s pred %s" %(count_index, seg, mem, cpu, score))
                
                sql = "insert into train_result values(%s, %s, %s, %s, %s, %s)" % (query_id, seg, cpu, mem, score, test_id)
                cur.execute(sql)
                #print(sql)
                count_index = 0   
    cur.execute("select * from train_result")
    for row in cur.fetchall():
        print(row)
myConnection.commit()
myConnection.close()