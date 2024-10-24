import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
from helpers import vn_processing as xt  # Giả sử vn_processing chứa hàm stepByStep

warnings.filterwarnings('ignore')

# Đọc dữ liệu từ tệp CSV vào DataFrame
data = pd.read_csv('data_cleaned/data_model.csv')

# Loại bỏ các hàng có giá trị NaN trong cột nhãn
data = data.dropna(subset=['Label'])

# Chia dữ liệu thành X và y
X = data['Comment Tokenize']
y = data['Label']

# Ánh xạ nhãn chuỗi sang số nguyên với ba nhãn: Tích cực, Tiêu cực, Bình thường
label_map = {
    'Tiêu cực': 0,
    'Bình thường': 1,
    'Tích cực': 2
}
y = y.map(label_map).dropna()

# Chia tập dữ liệu thành huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện TF-IDF Vectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=0.02, max_df=0.9)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Lưu mô hình TF-IDF
with open('models/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Khởi tạo các mô hình
xgb_model = XGBClassifier(objective='multi:softmax', num_class=len(label_map),
                           n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
svm_model = SVC(kernel='linear', probability=True, random_state=42)
logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                     max_iter=500, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Thêm mô hình Random Forest

# Huấn luyện từng mô hình và tính toán độ chính xác
models = {
    "XGBoost": xgb_model,
    "SVC": svm_model,
    "Logistic Regression": logistic_model,
    "Random Forest": rf_model
}

for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác của mô hình {model_name}: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=label_map.keys()))  # Báo cáo phân loại

# Voting Classifier kết hợp các mô hình trên
voting_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('svm', svm_model),
        ('logistic', logistic_model),
        ('rf', rf_model)  # Thêm mô hình Random Forest vào Voting
    ],
    voting='soft'  # 'soft' lấy xác suất, 'hard' lấy nhãn số đông
)

# Huấn luyện Voting Classifier
voting_model.fit(X_train_tfidf, y_train)

# Lưu mô hình Voting Classifier
with open('models/voting_model.pkl', 'wb') as f:
    pickle.dump(voting_model, f)

# Hàm dự đoán với mô hình Voting Classifier, chỉ nhận bình luận
def predict_comment(text):
    df = pd.DataFrame({'Comment': text})
    df['Comment Tokenize'] = df['Comment'].apply(xt.stepByStep)
    X_test = tfidf.transform(df['Comment Tokenize'])
    y_pred = voting_model.predict(X_test)

    # Ánh xạ nhãn về dạng chuỗi
    inv_label_map = {v: k for k, v in label_map.items()}
    df['Label'] = y_pred
    df['Label'] = df['Label'].map(inv_label_map)

    return df[['Comment', 'Label']]

# Thử nghiệm dự đoán với bình luận trong nhiều tình huống
test_comments = [
    "Không biết có nên quay lại không."
]

results = predict_comment(test_comments)
print("Kết quả dự đoán:")
print(results)

# Đánh giá mô hình Voting Classifier
y_pred = voting_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình Voting: {accuracy:.2f}")
print(classification_report(y_test, y_pred, target_names=label_map.keys()))  # Báo cáo phân loại
