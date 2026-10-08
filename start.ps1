Start-Process "cmd.exe" -ArgumentList "/c title MargDrishti Backend && python -m uvicorn backend.app.main:app --reload"
Start-Process "cmd.exe" -ArgumentList "/c title MargDrishti Frontend && cd frontend && npm run dev"
echo "Backend starting at http://127.0.0.1:8000"
echo "Frontend starting at http://localhost:5173"
