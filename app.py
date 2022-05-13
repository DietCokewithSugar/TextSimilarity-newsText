from flask import Flask, render_template, url_for, request
import gensim
import numpy as np
import jieba
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

stop_text = open('static/stop_list.txt', 'r')
stop_word = []
for line in stop_text:
    stop_word.append(line.strip())
TaggededDocument = gensim.models.doc2vec.TaggedDocument
model_dm = Doc2Vec.load("model_doc2vec_big")



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/get_corpus')
def get_corpus():
    with open("static/corpus_seg_old.txt", 'r') as doc:
        docs = doc.readlines()
    train_docs = []
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        length = len(word_list)
        word_list[length - 1] = word_list[length - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        train_docs.append(document)
    return train_docs

@app.route('/judge', methods=['POST'])
def judge():
    if request.method == 'POST':
        message = request.form['message']
        text_test = "当地时间12月30日，据英国广播公司报道，就在英国议会下院以448票巨大票差顺利通过英欧贸易协议后，首相约翰逊签署了刚刚从布鲁塞尔空运至伦敦的《英欧贸易与合作协议》。当天早些时候，欧洲理事会主席米歇尔、欧盟委员会主席冯德莱恩在布鲁塞尔签署了《英欧贸易与合作协议》。该协议随后空运英国伦敦，由英国首相约翰逊签署。签署后，该协议将在2021年1月1日开始临时实施。（原题为《英国首相约翰逊签署<英欧贸易与合作协议>》）(本文来自澎湃新闻，更多原创资讯请下载“澎湃新闻”APP)"
        text_cut = jieba.cut(message)
        text_raw = []
        for i in list(text_cut):
            text_raw.append(i)
        inferred_vector_dm = model_dm.infer_vector(text_raw)
        sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=5)

        x_train = get_corpus()

        i = 0
        textArray = [('未找到相似度高于75%的文章'), ('未找到相似度高于75%的文章'), ('未找到相似度高于75%的文章'), ('未找到相似度高于75%的文章'), ('未找到相似度高于75%的文章')]
        textCount = [0, 0, 0, 0, 0]
        textSim = [('0%'), ('0%'), ('0%'), ('0%'), ('0%')]

        for count, sim in sims:
            sentence = x_train[count]
            words = ''
            for word in sentence[0]:
                words = words + word
            if sim >= 0.75:
                textArray[i] = words
                textCount[i] = len(words)
                simsim = sim * 100
                textSim[i] = str(round(simsim,2)) + '%'
            i = i +1
            print(words, sim, len(sentence[0]), i)
    return render_template('result.html', textSim = textSim, textCount = textCount, textArray = textArray, message = message, rawLen = len(message))

if __name__ == '__main__':
	app.run(debug=True)