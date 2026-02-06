from __future__ import annotations

import uvicorn


def main() -> None:
    app_import = "visualcounter.api:create_app"
    uvicorn.run(app_import, factory=True, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
