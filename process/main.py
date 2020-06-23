import sys
import os
import io
import re
import nltk
import urllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm, datasets
from sklearn.metrics import plot_confusion_matrix


ps = PorterStemmer()
dir_path = os.path.dirname(os.path.realpath(__file__))


####################################################################################################
# Các hàm tiện ích
def get_text(file):
    read_file = io.open(file, 'r', encoding='utf-8')
    text = read_file.readlines()
    text = ' '.join(text)
    return text


def clean_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


def remove_special_character(text):
    string = re.sub('[^\w\s]', '', text)
    string = re.sub('\s+', ' ', string)
    string = string.strip()
    return string


def count_duplicate_words(words):
    hash = {}
    for word in words:
        if word not in hash:
            hash[word] = 1
        else:
            hash[word] += 1
    return hash


def bag_of_words(sentences):
    result = CountVectorizer()
    dense = result.fit_transform(sentences).todense()

    # print(result.fit_transform(sentences).todense())
    # print(result.vocabulary_)

    return dense


def tf_idf(sentences):
    tf = TfidfVectorizer(analyzer='word',
                         ngram_range=(1, 3), min_df=0, stop_words='english')
    tf_idf_matrix = tf.fit_transform(sentences)
    feature_names = tf.get_feature_names()
    dense = tf_idf_matrix.todense()

    # print('\n'.join(feature_names))
    # print(tf_idf_matrix)
    # print(dense)

    return dense


def write_file(file_name, text):
    with io.open(os.path.join(dir_path, 'output/{0}'.format(file_name)), 'w', encoding='utf-8') as file:
        file.write(text)
        print('Output to {0}'.format(file_name))


def write_dictionary_to_file(file_name, dictData):
    with io.open(os.path.join(dir_path, 'output/{0}'.format(file_name)), 'w', encoding='utf-8') as file:
        for key, value in dictData.items():
            file.write('{0: <15} : {1}\n'.format(key, value))
        print('Output to {0}'.format(file_name))


####################################################################################################
# Tiền xử lý
def pretreatment(path, file_name, my_stopwords, corpus):
    # Lấy văn bản
    text = get_text(path)
    text_cleaned = clean_html(text)

    # Tách câu
    sents = sent_tokenize(text_cleaned)
    # Loại bỏ kỹ tự đặc biết trong câu
    sents_cleaned = [remove_special_character(s) for s in sents]
    # Nối các câu lại thành text
    text_sents_join = ' '.join(sents_cleaned)

    # Tách từ
    words = word_tokenize(text_sents_join)
    # Đưa về dạng chữ thường
    words = [word.lower() for word in words]
    # Loại bỏ hư từ
    words = [word for word in words if word not in my_stopwords]

    # Chuẩn hoá từ
    words = [ps.stem(word) for word in words]

    # Tổng kết từ lặp
    words_summary = count_duplicate_words(words)

    # Nối words của tệp lại thành câu và thêm vào corpus
    joinedWords = ' '.join(set(words))
    corpus.append(joinedWords)

    # Xuất output
    file_name = os.path.splitext(file_name)[0]
    # 1) Xuất file tách html
    write_file('pretreat/{0}_text.txt'.format(file_name), text_cleaned)
    # 2) Xuất file tách câu
    write_file('pretreat/{0}_sentence.txt'.format(file_name),
               '\n'.join(sents_cleaned))
    # 3) Xuất file tách từ
    write_file('pretreat/{0}_word.txt'.format(file_name), '\n'.join(words))
    # 4) Xuất file tổng kết từ
    write_dictionary_to_file(
        'pretreat/{0}_summary_word.txt'.format(file_name), words_summary)


####################################################################################################
# Xử lý vector
def vectorizer(file_names, corpus, method):
    results = []
    chosen = ""

    if method == 1:
        chosen = "BoW"
        results = bag_of_words(corpus)

    if method == 2:
        chosen = "TF_IDF"
        results = tf_idf(corpus)

    arr_results = np.array(results)
    with io.open(os.path.join(dir_path, 'output/{0}.txt'.format(chosen)), 'w', encoding='utf-8') as file:
        for i in range(0, len(file_names)):
            num_str = ""
            if method == 1:
                num_str = ' '.join([str(num) for num in arr_results[i]])
            if method == 2:
                num_str = ' '.join([str(num).ljust(20, ' ')
                                    for num in arr_results[i]])

            row = "{0} {1: <10} {2}\n".format(
                str(i + 1), file_names[i], num_str)

            file.write(row)

        print('Output to {0}.txt'.format(chosen))

    return results


####################################################################################################
# Xử lý độ tương đồng
def similarity(vectorizer_result, method, corpus):
    results = []
    vectors = np.array(vectorizer_result)
    vector_method = ""

    if method == 1:
        vector_method = "BoW"
    else:
        vector_method = "TF_IDF"

    similarity_method = "CosSim"
    for i in range(0, len(vectors)):
        result = []
        for j in range(0, len(vectors)):
            result.append(1 - spatial.distance.cosine(vectors[i], vectors[j]))
        results.append(result)

    with io.open(os.path.join(dir_path, 'output/{0}_{1}.txt'.format(vector_method, similarity_method)), 'w', encoding='utf-8') as file:
        for i in range(0, len(results)):
            num_str = ""
            num_str = ' '.join([str(round(num, 3)).ljust(6, ' ')
                                for num in results[i]])
            row = "{0}\n".format(num_str)
            file.write(row)

    print('Output to {0}_{1}.txt'.format(vector_method, similarity_method))


####################################################################################################
# Phân chia dữ liệu train test từ kNN
def kNN(X, y):
    X = pd.DataFrame(X)
    y = pd.Series(y)
    print('\n>>>>> X')
    print(X)
    print('\n>>>>> y')
    print(y)

    # Phân chia dữ liệu train và test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.3)

    print('\n>>>>> X_train')
    print(X_train)
    print('\n>>>>> X_test')
    print(X_test)
    print('\n>>>>> y_train')
    print(y_train)
    print('\n>>>>> y_test')
    print(y_test)
    print('\n>>>>> Calculation <<<<<<')

    # Khai báo lớp KNN với k=10
    knn = KNeighborsClassifier(n_neighbors=10)
    # Huấn luyện
    knn.fit(X_train, y_train)

    # Tính độ chính xác
    accuracy = knn.score(X_test, y_test)

    # Tạo ma trận nhầm lẫn (confusion matrix)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # print('>>>>>> Delta (y_test - y_pred)')
    # print(set(y_test) - set(y_pred))

    # Tính toán độ đo precision
    precision = precision_score(y_test, y_pred, average='weighted')

    # Tính toán độ recall
    recall = recall_score(y_test, y_pred, average='weighted')

    # Tính toán dộ đo F1
    f1 = f1_score(y_test, y_pred, average='weighted')

    print('\n>>>>> accuracy')
    print(accuracy)
    print('\n>>>>> precision')
    print(precision)
    print('\n>>>>> recall')
    print(recall)
    print('\n>>>>> f1')
    print(f1)

    # Log ra file
    text = '>>> accuracy: {0}\n>>> precision: {1}\n>>> recall: {2}\n>>> f1: {3}'.format(
        accuracy, precision, recall, f1)
    write_file('calculation.txt', text)

    return [knn, X_train, X_test, y_train, y_test]


####################################################################################################
# Tính cross validation
def k_fold_cross_validation(classifier, X, y):
    # Huấn luyện với 5 folds
    cv_scores = cross_val_score(classifier, X, y, cv=5)

    # Độ chính xác từng fold
    print('\n>>>>> cv_scores')
    print(cv_scores)

    # Độ chính xác trung bình
    print('\n>>>>> cv_scores means')
    mean_cv_scores = np.mean(cv_scores)
    print('cv_scores mean: {}'.format(mean_cv_scores))

    # Log ra file
    text = '>>> cv_scores: {0}\n>>> cv_scores means: {1}'.format(
        cv_scores, mean_cv_scores)
    write_file('cross_validation.txt', text)


####################################################################################################
# Vẽ ma trận nhầm lẫn
def draw_confusion_matrix(classifier, X_train, X_test, y_train, y_test, labels):
    np.set_printoptions(precision=2)
    cm = []

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=labels,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        cm.append(title)
        cm.append(str(disp.confusion_matrix))

        print(title)
        print(disp.confusion_matrix)

    # Log ra file
    write_file('confusion_matrix.txt', '\n'.join(cm))

    plt.show()


####################################################################################################
# Hàm main
def main(argv):
    path = os.path.join(dir_path, 'input/')
    list_path = []
    file_names = []
    corpus = []

    # Declare labels and index
    labels = []
    index = []

    i = 0
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            labels.append(dir)
            for root, dirs, files in os.walk(path + dir):
                for file in files:
                    list_path.append(root + "/" + file)
                    file_names.append(file)
                    index.append(i)
            i += 1

    # print(labels)
    # print(index)

    # Khởi tạo stopwords
    my_stopwords = set(stopwords.words('english') + list(punctuation))

    # Tạo folder pretreat
    pretreat_path = os.path.join(dir_path, 'output/pretreat')
    if not os.path.exists(pretreat_path):
        os.mkdir(pretreat_path)

    # Tiền xử lý từ trong mỗi file và lấy corpus
    for i in range(0, len(list_path)):
        pretreatment(list_path[i], file_names[i], my_stopwords, corpus)

    # Xử lý vector
    method = -1
    while method not in [1, 2]:
        try:
            method = int(
                input("Chon phuong phap tinh (1. Bag Of Words, 2. TF_IDF): "))
        except ValueError:
            print('!!! Hay chon 1 hoac 2.')
            method = -1
    vectorizer_result = vectorizer(file_names, corpus, method)

    # Tính độ tương đồng
    print('Xu ly do tuong dong...')
    similarity(vectorizer_result, method, corpus)

    # Xử lý kNN
    X, y = vectorizer_result, index
    knn, X_train, X_test, y_train, y_test = kNN(X, y)

    # Thử nghiệm k-fold cross validation
    k_fold_cross_validation(knn, X, y)

    # Vẽ confusion matrix
    print()
    draw_confusion_matrix(knn, X_train, X_test, y_train, y_test, labels)

    print('\nExit!!!')


if __name__ == '__main__':
    main(sys.argv[1:])
