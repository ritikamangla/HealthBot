# /index.py
from flask import Flask, request, jsonify, render_template
import os
import dialogflow
import requests
import json
import pusher

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(silent=True)
    if data['queryResult']['queryText'] == 'no':
        #from database import r1

        report()

        file = open("Disease.txt", "r")
        p = file.readlines()
        print("This is th vaqlue"+p[0]+p[1]+p[2]+p[3]+p[4]+p[5]+p[6]+p[7])

        reply = {
            "fulfillmentText": "Ok. Here is your report.\n1."+p[0]+"2.\n\n"+p[1]+"3.\n\n"+p[2]+"4.\n\n"+p[3]+"5.\n\n"+p[4]+"6.\n\n"+p[5]+"7.\n\n"+p[6]+"8.\n\n"+p[7],

        }

        return jsonify(reply)

    else:
        reply = {
            "fulfillmentText": "Are you facing any other symptom?",
        }
        print(data['queryResult']['queryText'])
        file=open("Symptom.txt","a")
        file.write(data['queryResult']['queryText'])
        file.write("\n")
        file.close()
        return jsonify(reply)

def detect_intent_texts(project_id, session_id, text, language_code):
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(project_id, session_id)

        if text:
            text_input = dialogflow.types.TextInput(
                text=text, language_code=language_code)
            query_input = dialogflow.types.QueryInput(text=text_input)
            response = session_client.detect_intent(
                session=session, query_input=query_input)
            return response.query_result.fulfillment_text
@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form['message']
    project_id = os.getenv('DIALOGFLOW_PROJECT_ID')
    fulfillment_text = detect_intent_texts(project_id, "unique", message, 'en')
    response_text = { "message":  fulfillment_text }
    return jsonify(response_text)

def report():
    import numpy as np  # linear algebra
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
    from subprocess import check_output

    import os
    print(os.listdir('../input'))
    diff = pd.read_csv('../input/diffsydiw.csv')
    sym = pd.read_csv('../input/sym_t.csv')
    dia = pd.read_csv('../input/dia_t.csv')
    # print(sym.head())
    # dia['idnr'] = dia['_id'].convert_objects(convert_numeric=True)
    # print(dia.head())
    sd_diff = diff.merge(sym, left_on='syd', right_on='syd')
    #print(sd_diff.head())
    sd_diff=sd_diff.merge(dia, left_on='did', right_on='did')
    # print(sd_diff.head())

    from sklearn.preprocessing import LabelEncoder
    from tqdm import tqdm

    from scipy.sparse import coo_matrix, csr_matrix

    def read_data(filename):
        """ Reads in the last.fm dataset, and returns a tuple of a pandas dataframe
        and a sparse matrix of song/user/playcount """
        # read in triples of user/song/playcount from the input dataset
        data = pd.read_csv(filename,
                           usecols=[0, 1, 2],  # [36, 11, 10] vrk_pat_primkey,prd_atc_primkey,vdp_aantal
                           names=['user', 'song', 'plays'],
                           skiprows=1)  # [:1000000]   # user = patient, or prescriptionnr song=atc

        data = data.dropna(axis=0, how='any')  # drop nan
        data['plays'] = data['plays'] + 1
        # print(data.head())
        # map each song and user to a unique numeric value
        data['user'] = data['user'].astype("category")
        data['song'] = data['song'].astype("category")

        # create a sparse matrix of all the users/plays
        plays = coo_matrix((data['plays'].astype(float),
                            (data['song'].cat.codes.copy(),
                             data['user'].cat.codes.copy())))

        return data, plays, data.groupby(['song']).plays.sum(), data['user'].cat.codes.copy()

    data, matrix, songsd, user = read_data('../input/diffsydiw.csv')
    data.head(10)

    # user=symptom
    # sond=diagnose

    from sklearn.preprocessing import normalize

    def cosine(plays):
        normalized = normalize(plays)
        return normalized.dot(normalized.T)

    def bhattacharya(plays):
        plays.data = np.sqrt(plays.data)
        return cosine(plays)

    def ochiai(plays):
        plays = csr_matrix(plays)
        plays.data = np.ones(len(plays.data))
        return cosine(plays)

    def bm25_weight(data, K1=1.2, B=0.8):
        """ Weighs each row of the matrix data by BM25 weighting """
        # calculate idf per term (user)
        N = float(data.shape[0])
        idf = np.log(N / (1 + np.bincount(data.col)))

        # calculate length_norm per document (artist)
        row_sums = np.squeeze(np.asarray(data.sum(1)))
        average_length = row_sums.sum() / N
        length_norm = (1.0 - B) + B * row_sums / average_length

        # weight matrix rows by bm25
        ret = coo_matrix(data)
        ret.data = ret.data * (K1 + 1.0) / (K1 * length_norm[ret.row] + ret.data) * idf[ret.col]
        return ret

    def bm25(plays):
        plays = bm25_weight(plays)
        return plays.dot(plays.T)

    def get_largest(row, N=10):
        if N >= row.nnz:
            best = zip(row.data, row.indices)
        else:
            ind = np.argpartition(row.data, -N)[-N:]
            best = zip(row.data[ind], row.indices[ind])
        return sorted(best, reverse=True)

    def calculate_similar_artists(similarity, artists, artistid):
        neighbours = similarity[artistid]
        top = get_largest(neighbours)
        return [(artists[other], score, i) for i, (score, other) in enumerate(top)]

    # songsd = dict(enumerate(data['song'].cat.categories))
    user_count = data.groupby('user').size()
    # to_generate = sorted(list(songsd), key=lambda x: -user_count[x])

    similarity = bm25_weight(matrix)

    # print(sym)
    sym[sym['syd'].isin(list(songsd.index))]

    from scipy.sparse.linalg import svds

    Ur, Si, VTr = svds(bm25_weight(coo_matrix(matrix)), k=100)
    # print(Ur.shape, Si.shape, VTr.shape,user.shape,matrix.shape,data.shape,songsd.shape,user_count.shape)
    VTr = pd.DataFrame(VTr)

    from sklearn.metrics.pairwise import cosine_similarity
    Sddf = pd.DataFrame(cosine_similarity(Ur, VTr.T), columns=user_count.index, index=list(songsd.index))
    Sddf.to_csv('Sddf.csv')

    Sydi = pd.DataFrame(cosine_similarity(Ur, VTr.T))

    ###changes

    # booknr=13 #symptoom4
    # b='Headache'
    # print('Symptom',sym[sym['symptom']==b])

   # a = input("Enter your symptom:")
    file=open("Symptom.txt","r")
    x=file.readlines()
    file.close()
    print(x)
    r1=[]
    for j in range(len(x)):
        a=x[j]
        #print("symptom")
        #print(x[0])
        #print(a)
        data = pd.read_csv("sym_t.csv")
        c=0
        for i in data['symptom']:
            print(c)
            c += 1
            print(type(i))
            #print("i is"+i)
            if a == (i+"\n"):
                break

        print("Value of c")
        print(c)
        booknr = c  # symptoom4
        print('Symptom', sym[sym['syd'] == booknr])
        print('top 7 related disease probability')  # ,Sddf[booknr].sort_values(ascending=False))
        print()

        # print(sym.loc["symptom"]=="Headache")

        data = pd.read_csv("sym_t.csv", index_col="symptom")
       # print(data.loc["Headache"], ["syd"])
        #print("hi")
        lijst = Sddf[booknr].sort_values(ascending=False).index
        for xi in lijst[:4]:
            r1.append(dia[dia['did'] == xi].diagnose.values)
    #print(type(r1[0][0]))
    file = open("Disease.txt", "a")
    for j in range(len(r1)):
        file.write(r1[j][0])
        file.write("\n")
    file.close()

    #print(r1[0][0])

    '''lijst = list(lijst[:3])
    vectori = [0 for x in range(0, len(user_count))]
    for xi in lijst:
        # ii=songsd.index(xi)
        vectori += Sddf.loc[xi]
    vect = pd.DataFrame(vectori)
    vect.columns = ['para']
    vect = vect[vect > 1].dropna(axis=0)
    print('other symptoms', vect / 3)
    print("hi")
    print(sym[sym.syd.isin(list(vect.index))])
    print("BYE")'''


# run Flask app
if __name__ == "__main__":
    app.run()