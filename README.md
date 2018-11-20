Justine's Python Helper Tools
===

Usage
---

In python script (needs python 3):

```python3
import jc_tools
```

#### Example

```python3
from jc_tools.utils import group_by
```


Installation
---

Need to have read permissions to github repo and key access through command line:

```shell
$ pip3 install git+ssh://git@github.com/galvanic/jc_tools.git
```

On a remote server, need to install as `sudo`, and use environment variables to forward identity to github:

```shell
$ sudo -E pip3 install git+ssh://git@github.com/galvanic/jc_tools.git
```


Upgrade
---

```shell
$ sudo -E pip3 install git+ssh://git@github.com/galvanic/jc_tools.git --upgrade
```


Uninstall
---

```shell
$ sudo pip3 uninstall JustineTools
```


