# SimpleRAG: ローカル環境だけで動かせるRAGアプリケーション

このアプリは、シンプルなRetrieval-Augmented Generation (RAG) アプリケーションです。
ユーザーの質問に対して、ローカルのドキュメント(Docs配下)から関連情報を検索し、回答を生成します。
もしヒットしない場合、APIの設定をすれば外部のAPIサーバー（openAIなど）から回答を取ってきて回答することができます。


## 主な機能

- ローカルテキストファイル（.txt, .md）からの情報検索
- 日本語と英語の両方に対応した検索エンジン
- ドキュメントのアップロード機能
- ドキュメントの再読み込み機能
- テキストの自動チャンク分割
- TF-IDFとコサイン類似度に基づく検索
- 直接文字列マッチングによる検索強化
- OpenAI API（オプション）による回答生成

## プロジェクト構成

```
.
├── backend/                 # バックエンドAPI
│   └── app/
│       ├── docs/            # ドキュメント保存ディレクトリ
│       ├── config.py        # 設定ファイル
│       ├── simple_rag.py    # RAG実装
│       └── simple_server.py # FastAPI サーバー
└── frontend/                # フロントエンド (React)
```

## 技術スタック

- **バックエンド**: Python, FastAPI
- **RAG実装**: scikit-learn (TF-IDF), NLTK
- **外部API**: OpenAI API (オプション)
- **フロントエンド**: React

## セットアップと実行

### バックエンド

```bash
# 必要なパッケージのインストール
pip install fastapi uvicorn scikit-learn nltk openai

# サーバーの起動
cd backend
uvicorn app.simple_server:app --reload
```

サーバーは http://localhost:8000 で起動します。

### API エンドポイント

- `POST /ask`: 質問に対する回答を取得
- `GET /files`: 読み込まれているファイルの一覧を取得
- `POST /reload`: ドキュメントを再読み込み
- `POST /upload`: ファイルをアップロード
- `POST /settings/api`: OpenAI APIキーを設定
- `GET /settings/api/status`: API設定状況を確認
- `GET /search-test/{query}`: 検索機能のテスト

## 検索アルゴリズム

このシステムは複数の検索手法を組み合わせて、高精度な検索を実現しています：

1. **直接文字列マッチング**:
   - 完全一致: 検索クエリがそのままテキスト内に存在
   - 順序一致: 検索クエリの文字が順番通りに出現
   - 全文字一致: 検索クエリの全文字が含まれる（順不同）
   - 部分一致: 検索クエリの80%以上の文字が含まれる

2. **TF-IDFベクトル検索**:
   - 日本語対応のトークナイザーを使用
   - コサイン類似度による関連文書の特定
   - 複数のクエリ展開技術を使用（1文字ずつ、2文字ずつなど）

## 拡張機能

OpenAI APIキーを設定することで、ローカルドキュメントで十分な情報が見つからない場合に
外部のLLMを利用した回答生成が可能になります。

## ライセンス

MIT
