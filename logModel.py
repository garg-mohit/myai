from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest


logs = [
    "ERROR: Module A failed due to timeout",
    "WARNING: Memory usage exceeded",
    "INFO: Service restarted successfully",
    "ERROR: Disk read failure detected",
    "WARNING: Network latency detected",
    "INFO: Version Information",
    "INFO: Time taken since the time buffer",
    "ERROR: AVERAGE Time taken since the time buffer(.*)",
    "INFO: popData has not been called for",
    "INFO:Pushing new transaction id:",
    "INFO: Initialized a new transaction:",
    "INFO: Finalized the transaction:",
    "ERROR: Number of lines not same in",
    "INFO: captureVsSyncForStreamDiffMS:",
    "INFO: Started syncing transaction",
    "INFO: Ended syncing transaction",
    "WARNING: Sensor hit vs transaction start time difference is very less.",
    "INFO: stopConveyorTimeOut",
    "ERROR: in communication: Connection reset by peer",
    "ERROR: As GUI is disconnected, So asking peripheral thread to stop the conveyor.",
    "WARNING: ms m_maxTransDecisionDelayMS",
    "INFO: Expected time to show this buffered data in live screen is",
    "WARNING: this mismatch can cause delay in livescreen",
    "ERROR: GUI and DIU connection lost"
]
#labels = ["Error", "Warning", "Info", "Error", "Warning"]
labels = [each.split(':')[0] for each in logs]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(logs)
model = LogisticRegression()
model.fit(X, labels)
iso_forest = IsolationForest(contamination=0.1)
iso_forest.fit(X)


# Predict a new log entry
new_logs = [
    "Database connection lost",
    "CPU temperature too high",
    "Scheduled backup completed"
]

def analyze_logs(log_file, type=None):
    with (open(log_file, "r") as f):
        lines = f.readlines()
        for each in lines:
            new_X = vectorizer.transform([each])
            result = model.predict(new_X)
            anomaly_scores = iso_forest.predict(new_X)
            if type:
                if result[0] == type:
                    print(f"{each}:{result[0]}")
                    print(anomaly_scores)
            else:
                print(f"{each}:{result[0]}")
                print(anomaly_scores)



if __name__ == "__main__":
    log_file = "/home/gmohit/Kscan/issues/livescreen.log_01"
    analyze_logs(log_file, 'ERROR')