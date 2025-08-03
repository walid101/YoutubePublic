### /api is for any integrations/adapters with external microservices required for Manga Video Generation.

### Runner:
- API Test: python -m src.com.channels.manga.api.runner

### Tasks:
- ~~Make saving remotely more efficient. We only create folders UP TO the chapter level. From there we rely on filenames to detect pages. We will save chapter metadata and page metadata all in one chapter folder (chapter_x) then save that one folder all at once.~~

- ~~Investigate WARNING:src.com.channels.manga.api.MangaApi.MangaApi:Failed to remove temporary file page_21.json: [WinError 32] The process cannot access the file because it is being used by another process: 'page_21.json'. This causes some files to be saved locally, we dont want that.~~

- ~~Add a compress option to save image (i.e convert it to a compression format like jpg or jpeg). ~~