import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import joblib
import pickle
import warnings

warnings.filterwarnings('ignore')

nltk.download('punkt_tab')
class LogAnalyzer:
    def __init__(self):
        # Download NLTK resources if not already present
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('punkt')

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.log_df = None
        self.vectorizer = None
        self.model = None
        self.pca = None
        self.scaler = None

    def parse_logs(self, log_file, log_format=None):
        """
        Parse log file into a structured format

        log_format: regex pattern with named groups for timestamp, level, component, message
                   default will try to detect common formats
        """
        print(f"Parsing log file: {log_file}")

        # Common log format patterns
        if log_format is None:
            # Try to detect the format from first few lines
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as file:
                sample = ''.join([file.readline() for _ in range(5)])

            # Try to detect timestamp pattern
            if re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', sample):
                # Standard format with milliseconds
                log_format = r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+(?P<level>[A-Z]+)\s+\[(?P<component>[^\]]+)\]\s+(?P<message>.*)'
            elif re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', sample):
                # Standard format without milliseconds
                log_format = r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(?P<level>[A-Z]+)\s+\[(?P<component>[^\]]+)\]\s+(?P<message>.*)'
            # elif re.search(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', sample):
            #     # Apache-like format
            #     log_format = r'(?P<timestamp>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})\s+(?P<level>[A-Z]+)\s+(?P<message>.*)'
            elif re.search(r'<((\d){2}/(\d){2}/(\d){4}\s(\d){2}:(\d){2}:(\d){2})', sample):
                # kritiscan log format
                log_format = r'(?P<timestamp>\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2})\s+\d+\s+(?P<level>\d+)>\s+(?P<message>.*)'
            else:
                # Generic fallback format
                log_format = r'(?P<timestamp>[^\]]+)\s+(?P<level>[A-Z]+)\s+(?P<message>.*)'
                print("Warning: Could not detect log format. Using generic pattern.")

        logs = []
        multi_line_message = None
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                match = re.search(log_format, line)
                if match:
                    if multi_line_message is not None:
                        logs[-1]['message'] = logs[-1]['message'] + " " + " ".join(multi_line_message)
                        multi_line_message = None

                    log_entry = match.groupdict()
                    logs.append(log_entry)
                elif logs:  # This might be a continuation of the previous log message
                    if multi_line_message is None:
                        multi_line_message = [line.strip()]
                    else:
                        multi_line_message.append(line.strip())

        self.log_df = pd.DataFrame(logs)
        print(f"Parsed {len(self.log_df)} log entries")

        # Convert timestamp to datetime if present
        if 'timestamp' in self.log_df.columns:
            try:
                # Try to detect timestamp format
                sample_timestamp = self.log_df['timestamp'].iloc[0]

                if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', sample_timestamp):
                    self.log_df['timestamp'] = pd.to_datetime(self.log_df['timestamp'], format='%Y-%m-%d %H:%M:%S,%f')
                elif re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', sample_timestamp):
                    self.log_df['timestamp'] = pd.to_datetime(self.log_df['timestamp'], format='%Y-%m-%d %H:%M:%S')
                elif re.match(r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}', sample_timestamp):
                    self.log_df['timestamp'] = pd.to_datetime(self.log_df['timestamp'], format='%m/%d/%Y %H:%M:%S')
                else:
                    self.log_df['timestamp'] = pd.to_datetime(self.log_df['timestamp'], errors='coerce')
            except:
                print("Warning: Could not parse timestamp format")
                pass

        return self.log_df

    def preprocess_messages(self, column='message'):
        """Preprocess log messages for analysis"""
        if self.log_df is None:
            raise ValueError("No log data available. Parse logs first.")

        print("Preprocessing log messages...")

        # Make sure the column exists
        if column not in self.log_df.columns:
            raise ValueError(f"Column '{column}' not found in log data")

        # Lowercase
        self.log_df['processed_message'] = self.log_df[column].str.lower()

        # Remove IP addresses
        self.log_df['processed_message'] = self.log_df['processed_message'].str.replace(
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP_ADDR', regex=True)

        # Remove timestamps
        self.log_df['processed_message'] = self.log_df['processed_message'].str.replace(r'\d{2}:\d{2}:\d{2}(?:\.\d+)?',
                                                                                        'TIMESTAMP', regex=True)

        # Replace numbers
        self.log_df['processed_message'] = self.log_df['processed_message'].str.replace(r'\b\d+\b', 'NUM', regex=True)

        # Tokenize, remove stopwords and lemmatize
        def process_text(text):
            if not isinstance(text, str):
                return ""
            tokens = nltk.word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens if
                      word.isalpha() and word not in self.stop_words]
            return " ".join(tokens)

        self.log_df['processed_message'] = self.log_df['processed_message'].apply(process_text)

        return self.log_df

    def extract_features(self, method='tfidf', max_features=1000):
        """Extract features from log messages"""
        if self.log_df is None or 'processed_message' not in self.log_df.columns:
            raise ValueError("No preprocessed log data available")

        print(f"Extracting features using {method} method...")

        if method.lower() == 'count':
            self.vectorizer = CountVectorizer(max_features=max_features)
        else:
            self.vectorizer = TfidfVectorizer(max_features=max_features)

        X = self.vectorizer.fit_transform(self.log_df['processed_message'])
        feature_names = self.vectorizer.get_feature_names_out()

        # Convert to dense array if less than 10k samples for easier manipulation
        if X.shape[0] < 10000:
            X = X.toarray()

        print(f"Extracted {X.shape[1]} features")
        return X, feature_names

    def apply_clustering(self, X, method='kmeans', n_clusters=10, eps=0.5, min_samples=5):
        """Apply clustering to identify log patterns"""
        print(f"Applying {method} clustering...")

        if method.lower() == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
            self.log_df['cluster'] = self.model.fit_predict(X)
        elif method.lower() == 'dbscan':
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
            self.log_df['cluster'] = self.model.fit_predict(X)
        else:
            raise ValueError(f"Clustering method '{method}' not supported")

        # Get cluster statistics
        cluster_stats = self.log_df['cluster'].value_counts().sort_index()
        print("Cluster distribution:")
        for cluster_id, count in cluster_stats.items():
            print(f"Cluster {cluster_id}: {count} logs ({count / len(self.log_df) * 100:.2f}%)")

        return self.log_df['cluster']

    def visualize_clusters(self, X, perplexity=30):
        """Visualize clusters using PCA"""
        if self.log_df is None or 'cluster' not in self.log_df.columns:
            raise ValueError("Clustering not performed yet")

        print("Visualizing clusters using PCA...")

        # Normalize data
        self.scaler = StandardScaler()
        if isinstance(X, np.ndarray):
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.fit_transform(X.toarray())

        # Apply PCA for visualization
        self.pca = PCA(n_components=2, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)

        # Create visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.log_df['cluster'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Log Clusters Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.tight_layout()

        # Save and show figure
        plt.savefig('log_clusters.png')
        plt.close()
        print("Visualization saved as 'log_clusters.png'")

        return X_pca

    def analyze_clusters(self, top_n=5):
        """Analyze log clusters to identify patterns and anomalies"""
        if self.log_df is None or 'cluster' not in self.log_df.columns:
            raise ValueError("Clustering not performed yet")

        print("\nAnalyzing log clusters:")

        # Identify potential anomalies
        cluster_counts = self.log_df['cluster'].value_counts()
        small_clusters = cluster_counts[cluster_counts < len(self.log_df) * 0.01].index.tolist()

        # Get representative messages for each cluster
        cluster_insights = {}
        for cluster_id in sorted(self.log_df['cluster'].unique()):
            cluster_df = self.log_df[self.log_df['cluster'] == cluster_id]

            # Get most common words
            if len(cluster_df) > 0 and self.vectorizer is not None:
                cluster_text = " ".join(cluster_df['processed_message'].tolist())
                tokens = cluster_text.split()
                word_counts = pd.Series(tokens).value_counts()
                common_words = word_counts.head(10).index.tolist()
            else:
                common_words = []

            # Get sample messages
            sample_messages = cluster_df['message'].sample(min(top_n, len(cluster_df))).tolist()

            # Check if this is potentially an anomaly cluster
            is_anomaly = cluster_id in small_clusters or cluster_id == -1

            # Store insights
            cluster_insights[cluster_id] = {
                'size': len(cluster_df),
                'percentage': len(cluster_df) / len(self.log_df) * 100,
                'common_words': common_words,
                'sample_messages': sample_messages,
                'is_anomaly': is_anomaly
            }

            # Print insights
            print(f"\nCluster {cluster_id}:")
            print(f"Size: {len(cluster_df)} logs ({len(cluster_df) / len(self.log_df) * 100:.2f}%)")
            if is_anomaly:
                if cluster_id == -1:
                    print("Status: NOISE/OUTLIERS (DBSCAN noise points)")
                else:
                    print("Status: POTENTIAL ANOMALY (small cluster)")
            else:
                print("Status: NORMAL PATTERN")

            if common_words:
                print(f"Common words: {', '.join(common_words)}")

            print("Sample messages:")
            for i, msg in enumerate(sample_messages, 1):
                print(f"  {i}. {msg[:100]}..." if len(msg) > 100 else f"  {i}. {msg}")

            # Level distribution
            if 'level' in self.log_df.columns:
                level_counts = cluster_df['level'].value_counts()
                print("Log levels:")
                for level, count in level_counts.items():
                    print(f"  {level}: {count} ({count / len(cluster_df) * 100:.2f}%)")

        return cluster_insights

    def detect_anomalies(self, method='isolation_forest'):
        """Detect anomalies in log data"""
        if self.log_df is None:
            raise ValueError("No log data available")

        if method.lower() == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            if self.vectorizer is None:
                print("Feature extraction not performed yet. Extracting features...")
                X, _ = self.extract_features()
            else:
                X = self.vectorizer.transform(self.log_df['processed_message'])

            print("Detecting anomalies using Isolation Forest...")
            model = IsolationForest(random_state=42, contamination=0.05)
            self.log_df['anomaly'] = model.fit_predict(X)
            # Convert to boolean (1 for normal, -1 for anomaly)
            self.log_df['anomaly'] = self.log_df['anomaly'] == -1

            anomalies = self.log_df[self.log_df['anomaly']]
            print(f"Detected {len(anomalies)} anomalies ({len(anomalies) / len(self.log_df) * 100:.2f}% of logs)")

            return anomalies
        else:
            raise ValueError(f"Anomaly detection method '{method}' not supported")

    def temporal_analysis(self, window='1H'):
        """Analyze log patterns over time"""
        if self.log_df is None or 'timestamp' not in self.log_df.columns:
            raise ValueError("No timestamped log data available")

        print("Performing temporal analysis...")

        # Make sure timestamp is datetime
        if not pd.api.types.is_datetime64_dtype(self.log_df['timestamp']):
            print("Warning: Timestamp is not in datetime format")
            return None

        # Set timestamp as index
        time_df = self.log_df.set_index('timestamp')

        # Resample by time window
        log_counts = time_df.resample(window).size()

        # Plot log frequency
        plt.figure(figsize=(12, 6))
        log_counts.plot()
        plt.title(f'Log Frequency Over Time (Window: {window})')
        plt.xlabel('Time')
        plt.ylabel('Number of Logs')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('log_frequency.png')
        plt.close()
        print("Temporal analysis saved as 'log_frequency.png'")

        # If level information is available, plot by level
        if 'level' in self.log_df.columns:
            level_groups = time_df.groupby('level')
            plt.figure(figsize=(12, 6))

            for level, group in level_groups:
                level_counts = group.resample(window).size()
                level_counts.plot(label=level)

            plt.title(f'Log Levels Over Time (Window: {window})')
            plt.xlabel('Time')
            plt.ylabel('Number of Logs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('log_levels.png')
            plt.close()
            print("Log levels analysis saved as 'log_levels.png'")

        return log_counts

    def save_model(self, filename='log_analyzer_model.pkl'):
        """Save the trained model and associated data"""
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'pca': self.pca,
            'scaler': self.scaler
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filename}")

    def load_model(self, filename='log_analyzer_model.pkl'):
        """Load a previously trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']

        print(f"Model loaded from {filename}")

    def generate_report(self, output_file='log_analysis_report.html'):
        """Generate an HTML report of the log analysis results"""
        if self.log_df is None:
            raise ValueError("No log data available")

        # Basic log statistics
        total_logs = len(self.log_df)
        time_span = "N/A"
        if 'timestamp' in self.log_df.columns and pd.api.types.is_datetime64_dtype(self.log_df['timestamp']):
            time_span = f"{self.log_df['timestamp'].min()} to {self.log_df['timestamp'].max()}"

        level_stats = ""
        if 'level' in self.log_df.columns:
            level_counts = self.log_df['level'].value_counts()
            level_stats = "<h3>Log Levels</h3><ul>"
            for level, count in level_counts.items():
                level_stats += f"<li>{level}: {count} ({count / total_logs * 100:.2f}%)</li>"
            level_stats += "</ul>"

        component_stats = ""
        if 'component' in self.log_df.columns:
            component_counts = self.log_df['component'].value_counts().head(10)
            component_stats = "<h3>Top Components</h3><ul>"
            for component, count in component_counts.items():
                component_stats += f"<li>{component}: {count} ({count / total_logs * 100:.2f}%)</li>"
            component_stats += "</ul>"

        # Cluster information
        cluster_info = ""
        if 'cluster' in self.log_df.columns:
            cluster_counts = self.log_df['cluster'].value_counts().sort_index()
            cluster_info = "<h2>Log Clusters</h2><table border='1'><tr><th>Cluster</th><th>Count</th><th>Percentage</th><th>Status</th><th>Sample Message</th></tr>"

            for cluster_id, count in cluster_counts.items():
                cluster_df = self.log_df[self.log_df['cluster'] == cluster_id]
                sample_message = cluster_df['message'].iloc[0][:100] + "..." if len(
                    cluster_df['message'].iloc[0]) > 100 else cluster_df['message'].iloc[0]

                # Determine status
                if cluster_id == -1:
                    status = "NOISE/OUTLIERS"
                elif count < total_logs * 0.01:
                    status = "POTENTIAL ANOMALY"
                else:
                    status = "NORMAL PATTERN"

                cluster_info += f"<tr><td>{cluster_id}</td><td>{count}</td><td>{count / total_logs * 100:.2f}%</td><td>{status}</td><td>{sample_message}</td></tr>"

            cluster_info += "</table>"

        # Anomaly information
        anomaly_info = ""
        if 'anomaly' in self.log_df.columns:
            anomalies = self.log_df[self.log_df['anomaly']]
            anomaly_info = f"<h2>Anomalies</h2><p>{len(anomalies)} anomalies detected ({len(anomalies) / total_logs * 100:.2f}% of logs)</p>"

            if len(anomalies) > 0:
                anomaly_info += "<h3>Sample Anomalies</h3><ul>"
                for _, row in anomalies.head(10).iterrows():
                    message = row['message'][:100] + "..." if len(row['message']) > 100 else row['message']
                    anomaly_info += f"<li>{message}</li>"
                anomaly_info += "</ul>"

        # Build HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Log Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: s30px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                h3 {{ color: #2980b9; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .warning {{ color: #e74c3c; }}
                .info {{ color: #2980b9; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Log Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Overview</h2>
            <p>Total Logs: {total_logs}</p>
            <p>Time Span: {time_span}</p>

            {level_stats}
            {component_stats}

            {cluster_info}
            {anomaly_info}

            <h2>Visualizations</h2>
            <h3>Cluster Visualization</h3>
            <img src="log_clusters.png" alt="Log Clusters">

            <h3>Temporal Analysis</h3>
            <img src="log_frequency.png" alt="Log Frequency">
        </body>
        </html>
        """

        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"Analysis report generated: {output_file}")
        return output_file


def main(log_file_path):
    """Main function for log analysis pipeline"""
    analyzer = LogAnalyzer()

    # Parse logs
    analyzer.parse_logs(log_file_path)

    # Preprocess
    analyzer.preprocess_messages()

    # Extract features
    X, feature_names = analyzer.extract_features(method='tfidf', max_features=1000)

    # Clustering
    analyzer.apply_clustering(X, method='kmeans', n_clusters=10)

    # Visualize
    analyzer.visualize_clusters(X)

    # Analyze clusters
    analyzer.analyze_clusters()

    # Temporal analysis
    analyzer.temporal_analysis(window='1H')

    # Generate report
    analyzer.generate_report()

    # Save model
    analyzer.save_model()

    print("\nLog analysis complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python log_analyzer.py <log_file_path>")
        sys.exit(1)

    log_file_path = sys.argv[1]
    main(log_file_path)
