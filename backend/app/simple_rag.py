import os
import pathlib
import nltk
import numpy as np
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import DOCS_DIR

# ロギングの設定
logger = logging.getLogger(__name__)

# 必要なNLTKデータをダウンロード
logger.info("NLTKリソースをダウンロード中...")
nltk.download('punkt', quiet=True)

def tokenize_japanese(text):
    """
    日本語テキストを文字単位で分割する簡易トークナイザー
    
    英数字はそのままの単語として扱い、日本語は一文字ずつ分解します。
    これにより日本語の部分一致検索が可能になります。
    
    Args:
        text (str): 分割するテキスト
    
    Returns:
        list: トークンのリスト
    """
    # スペースで区切られた単語がある場合
    if ' ' in text:
        tokens = []
        for word in text.split():
            # 英数字はそのまま
            if word.isascii():
                tokens.append(word)
            else:
                # 日本語は文字単位に分解
                tokens.extend(list(word))
        return tokens
    
    # 単語の区切りがない場合は文字単位に分解
    result = []
    ascii_part = ""
    
    for char in text:
        if char.isascii():
            ascii_part += char
        else:
            # 前の英数字パートがあれば追加
            if ascii_part:
                result.append(ascii_part)
                ascii_part = ""
            # 日本語文字を追加
            result.append(char)
    
    # 最後の英数字パートがあれば追加
    if ascii_part:
        result.append(ascii_part)
    
    return result


class SimpleRAG:
    """
    シンプルなRetrieval-Augmented Generation (RAG)システム
    
    TF-IDFとコサイン類似度を使用して、ユーザークエリに関連する文書を検索し、
    回答を生成します。日本語と英語の両方に対応しています。
    """
    
    def __init__(self, exclude_files=None):
        """
        RAGシステムを初期化
        
        Args:
            exclude_files (list, optional): 読み込み対象から除外するファイル名のリスト
        """
        logger.info("SimpleRAGを初期化中...")
        self.documents = []          # ドキュメントのチャンク（テキスト断片）
        self.document_paths = []     # 各チャンクの元ファイルパス
        
        # 日本語対応のベクトライザー
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize_japanese,  # 日本語対応のトークナイザー
            stop_words=None,              # ストップワードなし（日本語対応）
            analyzer='word',              # 単語単位で分析
            ngram_range=(1, 2)            # 単語と2gramの両方を使用
        )
        self.doc_vectors = None
        
        # 除外ファイルリスト
        self.exclude_files = exclude_files or []
        
        # 類似度の閾値
        self.similarity_threshold = 0.05
        
        # 直接文字列検索を常に優先
        self.always_use_direct_match = True
        
        # 初期ドキュメントの読み込み
        self.load_documents()
        
    def load_documents(self):
        """
        docsディレクトリから文書を読み込み、チャンクに分割してベクトル化
        """
        # 既存のドキュメントとベクトルをクリア
        self.documents = []
        self.document_paths = []
        self.doc_vectors = None
        
        logger.info(f"ドキュメントを読み込み中... パス: {DOCS_DIR}")
        
        if not os.path.exists(DOCS_DIR):
            logger.error(f"ドキュメントディレクトリが見つかりません: {DOCS_DIR}")
            return
        
        # ディレクトリ内のファイル一覧
        all_files = list(pathlib.Path(DOCS_DIR).glob("**/*"))
        logger.info(f"ディレクトリ内のファイル数: {len(all_files)}")
        
        file_count = 0
            
        for p in pathlib.Path(DOCS_DIR).glob("**/*"):
            # 除外ファイルはスキップ
            if p.name in self.exclude_files:
                logger.info(f"ファイルを除外: {p}")
                continue
                
            # テキストファイルのみ処理
            if p.suffix.lower() in {".txt", ".md"}:
                file_count += 1
                logger.info(f"ファイルを処理中: {p}")
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # 文書をチャンクに分割
                        chunks = self._simple_split_text(content)
                        logger.info(f"ファイル {p} を {len(chunks)} チャンクに分割しました")
                        
                        # 各チャンクを追加
                        chunk_count = 0
                        for chunk in chunks:
                            if len(chunk.strip()) > 5:  # 短すぎるチャンクは無視
                                self.documents.append(chunk)
                                self.document_paths.append(str(p))
                                chunk_count += 1
                        
                        logger.info(f"追加されたチャンク数: {chunk_count}")
                        
                except Exception as e:
                    logger.error(f"ファイル読み込みエラー {p}: {e}")
        
        logger.info(f"処理されたファイル数: {file_count}")
                    
        # 文書がある場合はベクトル化
        if self.documents:
            logger.info(f"ドキュメントをベクトル化... {len(self.documents)}個のチャンク")
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)
            logger.info(f"ベクトル化完了。{len(self.documents)}個のドキュメントチャンクが利用可能")
        else:
            logger.warning(f"ドキュメントが見つかりません: {DOCS_DIR}")
    
    def reload_documents(self, exclude_files=None):
        """
        ドキュメントを再読み込みする
        
        Args:
            exclude_files (list, optional): 除外するファイル名のリスト
            
        Returns:
            int: 読み込まれたドキュメントチャンクの数
        """
        if exclude_files is not None:
            self.exclude_files = exclude_files
        self.load_documents()
        return len(self.documents)
    
    def _simple_split_text(self, text, chunk_size=500):
        """
        テキストを適切なサイズのチャンクに分割
        
        Args:
            text (str): 分割するテキスト
            chunk_size (int): チャンクの最大サイズ
            
        Returns:
            list: チャンクのリスト
        """
        # 段落で分割
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # 段落区切りがない場合は文単位で分割
        if not chunks:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        # それでもチャンクがない場合は単純に文字数で分割
        if not chunks:
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i+chunk_size])
                
        logger.info(f"テキストを{len(chunks)}個のチャンクに分割")
        return chunks
    
    def get_loaded_files(self):
        """
        読み込まれているファイルの一覧を返す
        
        Returns:
            list: ファイルパスのリスト
        """
        return list(set(self.document_paths))
    
    def _contains_sequence(self, text, char_list):
        """
        文字列内に指定された文字のシーケンスが含まれるか確認
        
        Args:
            text (str): 検索対象テキスト
            char_list (list): 検索する文字のリスト
            
        Returns:
            bool: 文字シーケンスが含まれる場合はTrue
        """
        if not char_list:
            return True
            
        # 空白文字を除外
        char_list = [c for c in char_list if c.strip()]
        if not char_list:
            return True
            
        # 順番通りに文字が含まれるか確認
        pos = 0
        for char in char_list:
            pos = text.find(char, pos)
            if pos == -1:
                return False
            pos += 1  # 次の位置から検索
        return True
    
    def query(self, query_text, top_k=5):
        """
        クエリに基づいて最も関連性の高い文書チャンクを返す
        
        複数の検索手法（直接文字列マッチング、文字順序検索、ベクトル検索）を
        組み合わせて、関連性の高い文書を見つけます。
        
        Args:
            query_text (str): 検索クエリ
            top_k (int): 返す結果の最大数
            
        Returns:
            list: 検索結果のリスト（各結果は辞書形式）
        """
        # ドキュメントのチェック
        if not self.documents or self.doc_vectors is None:
            logger.warning("ドキュメントが読み込まれていません")
            return "ドキュメントが読み込まれていません。"
        
        logger.info(f"検索クエリ: '{query_text}'")
        
        # 1. 直接文字列検索（最も優先度高）
        direct_matches = self._direct_string_search(query_text)
        
        # 直接マッチングがあればそれを優先
        if direct_matches and self.always_use_direct_match:
            logger.info(f"直接マッチング結果: {len(direct_matches)}件")
            sorted_results = sorted(direct_matches, key=lambda x: x["similarity"], reverse=True)
            return sorted_results[:top_k]
        
        # 2. ベクトル検索（バックアップ）
        vector_matches = self._vector_search(query_text, top_k)
        
        # 結果を統合
        all_results = vector_matches.copy()
        
        # 重複排除と結果のソート
        unique_results = self._deduplicate_results(all_results)
        sorted_results = sorted(unique_results.values(), key=lambda x: x["similarity"], reverse=True)
        
        # 結果がない場合の再試行
        if not sorted_results:
            return self._fallback_search(query_text, top_k)
        
        logger.info(f"クエリ「{query_text}」に対して{len(sorted_results)}件の結果が見つかりました")
        return sorted_results[:top_k]
    
    def _direct_string_search(self, query_text):
        """
        直接文字列マッチングによる検索
        
        Args:
            query_text (str): 検索クエリ
            
        Returns:
            list: マッチした結果のリスト
        """
        direct_matches = []
        
        for i, doc in enumerate(self.documents):
            doc_content = doc.lower()
            source_path = self.document_paths[i]
            
            # 完全一致（最も優先度高）
            if query_text.lower() in doc_content:
                direct_matches.append({
                    "content": doc,
                    "similarity": 0.95,
                    "source": source_path,
                    "match_type": "direct_exact"
                })
                logger.info(f"完全一致検出: {source_path}")
                continue
            
            # 文字順序一致
            query_chars = list(query_text.lower())
            if self._contains_sequence(doc_content, query_chars):
                direct_matches.append({
                    "content": doc,
                    "similarity": 0.90,
                    "source": source_path,
                    "match_type": "direct_sequence"
                })
                logger.info(f"文字列順序一致検出: {source_path}")
                continue
            
            # 全文字一致（順不同）
            if all(char in doc_content for char in query_chars if char.strip()):
                direct_matches.append({
                    "content": doc,
                    "similarity": 0.85,
                    "source": source_path,
                    "match_type": "direct_chars"
                })
                logger.info(f"文字一致検出: {source_path}")
                continue
            
            # 部分文字一致（80%以上）
            matches = sum(1 for char in query_chars if char in doc_content and char.strip())
            if matches >= max(2, len(query_chars) * 0.8):
                direct_matches.append({
                    "content": doc,
                    "similarity": 0.80,
                    "source": source_path,
                    "match_type": "direct_partial"
                })
                logger.info(f"部分一致検出: {source_path} ({matches}/{len(query_chars)}文字一致)")
                continue
        
        return direct_matches
    
    def _vector_search(self, query_text, top_k):
        """
        TF-IDFベクトルとコサイン類似度を使用した検索
        
        Args:
            query_text (str): 検索クエリ
            top_k (int): 返す結果の最大数
            
        Returns:
            list: マッチした結果のリスト
        """
        # 検索クエリの拡張
        search_queries = [query_text]
        
        # 短い単語の場合は閾値を下げる
        if len(query_text) <= 6:
            search_queries.append(f" {query_text} ")
            temp_threshold = max(0.01, self.similarity_threshold * 0.3)
        else:
            temp_threshold = self.similarity_threshold
        
        # 日本語の場合、文字単位に分解
        if ' ' not in query_text and len(query_text) > 2:
            # 1文字ずつに分解
            expanded_query_chars = ' '.join(list(query_text))
            search_queries.append(expanded_query_chars)
            
            # 2文字ずつに分解
            expanded_query_bigrams = ' '.join([query_text[i:i+2] for i in range(0, max(1, len(query_text)-1))])
            search_queries.append(expanded_query_bigrams)
        
        # ベクトル検索実行
        vector_matches = []
        for search_query in search_queries:
            query_vector = self.vectorizer.transform([search_query])
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # 最も類似度の高いtop_k個のインデックスを取得
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            for i in top_indices:
                if similarities[i] >= temp_threshold:
                    vector_matches.append({
                        "content": self.documents[i],
                        "similarity": similarities[i],
                        "source": self.document_paths[i],
                        "match_type": "vector"
                    })
        
        return vector_matches
    
    def _deduplicate_results(self, results):
        """
        検索結果から重複を除去
        
        Args:
            results (list): 検索結果のリスト
            
        Returns:
            dict: 重複を除去した結果（キーは文書の一意識別子）
        """
        unique_results = {}
        for result in results:
            key = result["source"] + ":" + result["content"][:50]
            if key not in unique_results or result["similarity"] > unique_results[key]["similarity"]:
                unique_results[key] = result
        
        return unique_results
    
    def _fallback_search(self, query_text, top_k):
        """
        通常検索で結果が得られなかった場合のフォールバック検索
        
        Args:
            query_text (str): 検索クエリ
            top_k (int): 返す結果の最大数
            
        Returns:
            list: 検索結果のリスト
        """
        logger.info("通常検索で結果が見つからず、フォールバック検索を実行")
        
        # ドキュメントを再読み込み
        self.load_documents()
        
        # 直接一致検索
        retry_results = []
        for i, doc in enumerate(self.documents):
            if query_text.lower() in doc.lower():
                retry_results.append({
                    "content": doc,
                    "similarity": 0.8,
                    "source": self.document_paths[i],
                    "match_type": "retry"
                })
        
        if retry_results:
            logger.info(f"再試行で{len(retry_results)}件の結果が見つかりました")
            return retry_results[:top_k]
        
        # 非常に低い閾値でのベクトル検索
        last_resort_threshold = 0.001
        query_vector = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        for i in top_indices:
            if similarities[i] >= last_resort_threshold:
                retry_results.append({
                    "content": self.documents[i],
                    "similarity": similarities[i],
                    "source": self.document_paths[i],
                    "match_type": "last_resort"
                })
        
        return retry_results[:top_k]
    
    def get_answer(self, query_text):
        """
        クエリに対する回答を生成
        
        ローカルドキュメントで十分な情報が見つからない場合は
        LLM API（OpenAI）を利用します。
        
        Args:
            query_text (str): ユーザーの質問
            
        Returns:
            str: 質問に対する回答
        """
        logger.info(f"回答を生成中: {query_text}")
        relevant_docs = self.query(query_text)
        
        # ローカルドキュメントで十分な情報が見つからない場合
        if not relevant_docs or isinstance(relevant_docs, str) or len(relevant_docs) == 0:
            logger.info("ローカルで十分な情報が見つかりませんでした。LLM APIを利用します。")
            return self._get_llm_answer(query_text)
        
        # 類似度が低い場合もLLMを使用
        if all(doc["similarity"] < 0.15 for doc in relevant_docs):
            logger.info("類似度が低いためLLM APIを利用します。")
            return self._get_llm_answer(query_text)
        
        # ファイル単位でグループ化
        grouped_docs = {}
        for doc in relevant_docs:
            file_path = doc["source"]
            if file_path not in grouped_docs or doc["similarity"] > grouped_docs[file_path]["similarity"]:
                grouped_docs[file_path] = {
                    "content": doc["content"],
                    "similarity": doc["similarity"]
                }
        
        # 類似度順にソート
        sorted_docs = sorted(grouped_docs.items(), key=lambda x: x[1]["similarity"], reverse=True)
        
        # 関連情報を結合
        context = "\n\n".join([f"From {file_path}:\n{doc['content']}" for file_path, doc in sorted_docs])
        
        # 回答生成
        answer = f"質問: {query_text}\n\n関連情報:\n{context}\n\n"
        answer += "以上の情報に基づくと、"
        
        # 最も関連性の高い情報源を参照
        if sorted_docs[0][1]["similarity"] > self.similarity_threshold:
            top_source = os.path.basename(sorted_docs[0][0])
            answer += f"「{top_source}」から得られた情報が最も関連性が高いです。"
        else:
            answer += "十分に関連性の高い情報は見つかりませんでした。"
            
        return answer
    
    def _get_llm_answer(self, query_text):
        """
        LLM APIを使用して回答を生成
        
        Args:
            query_text (str): ユーザーの質問
            
        Returns:
            str: LLMからの回答
        """
        try:
            # API利用可能かチェック
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "ローカルデータベースに十分な情報がなく、LLM APIのキーも設定されていません。APIキーを設定してください。"
                
            # OpenAI APIを使用
            try:
                import openai
                openai.api_key = api_key
                
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "あなたは質問に正確に答える日本語アシスタントです。"},
                        {"role": "user", "content": query_text}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                llm_response = response.choices[0].message.content
                return f"ローカルデータベースに十分な情報がなかったため、外部知識を利用しました。\n\n{llm_response}"
                
            except ImportError:
                logger.error("openaiモジュールがインストールされていません")
                return "openai モジュールがインストールされていません。pip install openai を実行してください。"
                
        except Exception as e:
            logger.error(f"LLM API呼び出しエラー: {str(e)}")
            return f"申し訳ありません。ローカルデータベースに情報が見つからず、外部APIの呼び出しにも失敗しました。\nエラー: {str(e)}"

# シングルトンインスタンス
rag = SimpleRAG() 