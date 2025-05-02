from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import shutil
from typing import Optional, List, Dict, Any
from app.config import DOCS_DIR
from app.simple_rag import rag

# =====================================
# ロギング設定
# =====================================
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "rag_app.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"アプリケーション起動 - ログファイル: {log_file}")
logger.info(f"ドキュメントディレクトリ: {DOCS_DIR}")

# =====================================
# APIモデル定義
# =====================================
class Question(BaseModel):
    """質問リクエストモデル"""
    query: str

class ExcludeFilesRequest(BaseModel):
    """ファイル除外リクエストモデル"""
    exclude_files: List[str] = []

class APISettings(BaseModel):
    """API設定リクエストモデル"""
    api_key: str

# =====================================
# FastAPIアプリケーション
# =====================================
app = FastAPI(
    title="SimpleRAG API",
    description="シンプルなRetrieval-Augmented Generation (RAG) システムのAPI",
    version="1.0.0"
)

# CORSミドルウェア設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では特定のオリジンに制限すべき
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# ヘルパー関数
# =====================================
def check_docs_dir() -> None:
    """ドキュメントディレクトリが存在するか確認"""
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)
        logger.info(f"ドキュメントディレクトリを作成しました: {DOCS_DIR}")

# =====================================
# イベントハンドラー
# =====================================
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時に実行"""
    logger.info("RAGシステムの初期化中...")
    
    # ドキュメントディレクトリが存在するか確認
    check_docs_dir()
    
    # RAGシステムの状態確認
    if hasattr(rag, 'documents') and rag.documents:
        doc_count = len(rag.documents)
        logger.info(f"RAGシステムは既に初期化されています。ドキュメント数: {doc_count}")
    else:
        logger.warning("RAGシステムのドキュメントが存在しません。docs ディレクトリを確認してください。")

# =====================================
# APIエンドポイント
# =====================================
@app.post("/ask", response_model=Dict[str, str])
async def ask(question: Question):
    """
    質問に対する回答を生成
    
    質問に関連するドキュメントを検索し、回答を生成します。
    関連情報が見つからない場合、設定されていればLLM APIを使用します。
    """
    try:
        logger.info(f"質問を受信: {question.query}")
        answer = rag.get_answer(question.query)
        logger.info("回答を生成しました")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        return {"answer": f"エラーが発生しました: {str(e)}"}

@app.get("/files", response_model=Dict[str, Any])
async def get_files():
    """
    読み込まれているファイルの一覧を取得
    
    - all_files: ドキュメントディレクトリ内のすべてのファイル
    - loaded_files: 実際に読み込まれているファイル
    - excluded_files: 除外されているファイル
    """
    try:
        check_docs_dir()
        
        files = []
        for p in os.listdir(DOCS_DIR):
            if p.endswith('.txt') or p.endswith('.md'):
                files.append(p)
        
        loaded_files = rag.get_loaded_files()
        
        return {
            "all_files": files,
            "loaded_files": [os.path.basename(file) for file in loaded_files],
            "excluded_files": rag.exclude_files
        }
    except Exception as e:
        logger.error(f"ファイル一覧取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload", response_model=Dict[str, Any])
async def reload_documents(request: Optional[ExcludeFilesRequest] = None):
    """
    ドキュメントを再読み込み
    
    除外ファイルリストを更新して、ドキュメントを再読み込みします。
    """
    try:
        exclude_files = request.exclude_files if request and request.exclude_files else None
        doc_count = rag.reload_documents(exclude_files)
        return {
            "status": "success", 
            "document_count": doc_count, 
            "excluded_files": rag.exclude_files
        }
    except Exception as e:
        logger.error(f"ドキュメント再読み込みエラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=Dict[str, Any])
async def upload_file(file: UploadFile = File(...)):
    """
    ファイルをアップロード
    
    ドキュメントファイル（.txtまたは.md）をアップロードして、自動的に再読み込みします。
    """
    try:
        check_docs_dir()
        
        # ファイル形式チェック
        if not (file.filename.endswith('.txt') or file.filename.endswith('.md')):
            raise HTTPException(
                status_code=400, 
                detail="サポートされているファイル形式は .txt と .md のみです"
            )
        
        file_path = os.path.join(DOCS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        doc_count = rag.reload_documents()
        
        return {
            "status": "success", 
            "filename": file.filename, 
            "document_count": doc_count,
            "message": f"ファイル {file.filename} がアップロードされ、ドキュメントが再読み込みされました"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ファイルアップロードエラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/settings/api", response_model=Dict[str, str])
async def update_api_settings(settings: APISettings):
    """
    APIキーを設定
    
    OpenAI APIキーを環境変数に設定します。
    """
    try:
        os.environ["OPENAI_API_KEY"] = settings.api_key
        logger.info("OpenAI APIキーが設定されました")
        return {"status": "success", "message": "APIキーが設定されました"}
    except Exception as e:
        logger.error(f"API設定エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/settings/api/status", response_model=Dict[str, str])
async def check_api_settings():
    """
    APIキーが設定されているか確認
    """
    api_key = os.getenv("OPENAI_API_KEY")
    return {
        "status": "configured" if api_key else "not_configured",
        "message": "APIキーが設定されています" if api_key else "APIキーが設定されていません"
    }

@app.get("/search-test/{query}", response_model=Dict[str, Any])
async def search_test(query: str):
    """
    検索機能のテスト用エンドポイント（デバッグ用）
    
    実際の検索処理をテストし、詳細な結果を返します。
    """
    logger.info(f"検索テスト: {query}")
    try:
        # クエリの実行
        results = rag.query(query)
        
        if isinstance(results, str):
            return {"error": results}
            
        # 結果の整形
        formatted_results = []
        for r in results:
            formatted_results.append({
                "content": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                "similarity": float(r["similarity"]),
                "source": os.path.basename(r["source"]),
                "match_type": r.get("match_type", "unknown")
            })
            
        return {
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
    except Exception as e:
        logger.error(f"検索テストエラー: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================
# メイン実行
# =====================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 