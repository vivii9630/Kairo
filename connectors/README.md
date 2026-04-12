# kairo-connectors

Adapters for external data sources (S3, Notion, web, Slack, ...). Each
connector opts into its own dependencies, so you only install what you use.

```python
from kairo_connectors import BaseConnector
# concrete connectors land in kairo_connectors.s3, .notion, .web, .slack
```

Install a specific connector:

```bash
pip install -e "./connectors[s3]"
pip install -e "./connectors[notion]"
pip install -e "./connectors[web]"
```
