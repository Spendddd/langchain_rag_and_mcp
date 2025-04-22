# leedcode刷题

进度: 100%

# 算法学习

双链表：collections.deque()类

数组/列表：list()函数、[]（类）

# 哈希表

自己实现哈希表用拉链算法，哈希函数一般取素数的模，素数一般取1009或者2069

拉链算法中，表格使用[[]]实现哈希表，python中list可以使用.pop(index)或者.remove([key, value])实现删除，使用.append([key, value])插入新键值对，用in判断是否存在键值对

https://blog.csdn.net/xq151750111/article/details/129740670

`collections.Counter()`类是`dict`类的一个子类，用于计数可哈希对象。它是一个集合，其中元素存储为字典键，它们的计数存储为字典的值。

## 例题

1. 存在重复元素 II
    
    题目描述：给你一个整数数组 nums 和一个整数 k ，判断数组中是否存在两个不同的索引 i 和 j ，满足 nums[i] == nums[j] 且 abs ( i − j ) < = k 。如果存在，返回 true；否则，返回 false 。
    
    解法1：哈希字典
    
    ```python
    # 找到重复元素和其索引
    class Solution:
        def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
            hash_map = {}
            for i in range(len(nums)):
                # 已经存在重复的情况
                if nums[i] in hash_map and abs(i - hash_map[nums[i]]) <= k:
                    return True
                else:
                    hash_map[nums[i]] = i
            return False
    ```
    
    解法2：哈希集合
    
    ```python
    # 维护一个长度为 k 的集合，相当于大小为k的滑动窗口
    class Solution:
        def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
            hash_set = set()
            for i in range(len(nums)):
                # 存在重复元素
                if nums[i] in hash_set:
                    return True
                hash_set.add(nums[i])
                # 及时删除超出数组长度的元素
                if len(hash_set) > k:
                    hash_set.remove(nums[i - k])
            return False
    
    ```
    
2. 存在重复元素 III
    
    题目描述：给你一个整数数组 nums 和两个整数 k 和 t 。请你判断是否存在 两个不同下标 i 和 j，使得 abs(nums[i] - nums[j]) <= t，同时又满足abs(i−j)<=k。如果存在则返回 true，不存在返回 false。
    
    ```python
    class Solution:
        def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
            bucket_dict = dict()
            for i in range(len(nums)):
                # 将nums[i]划分到大小为 t + 1 的不同桶中，分桶操作
                num = nums[i] // (t + 1)
                # print(num)
    
                # 如果桶中已经有元素，有相同的分桶结果，表示存在相同元素
                if num in bucket_dict:
                    return True
                # 将 nums[i] 放入桶中
                bucket_dict[num] = nums[i]
                # print(bucket_dict)
    
                # 判断左侧桶是否满足条件
                if (num - 1) in bucket_dict and abs(bucket_dict[num - 1] - nums[i]) <= t:
                    return True
                # 判断右侧桶是否满足条件
                if (num + 1) in bucket_dict and abs(bucket_dict[num + 1] - nums[i]) <= t:
                    return True
                # 将i-k之前的旧桶清除，因为之前的桶已经不满足条件了，相当于维护大小为k的滑动窗口，下一个i会+1，如果代码到这一步说明前面都没有找到下标从i-k到i-1中满足要求的元素，所以可以放心删除当前下标最小的nums[i-k]对应的桶【只有这一步控制下标范围这一个条件】，上面的全部判断都是基于【已经满足下标条件】的前提下进行对值条件的判断
                if i >= k:
                    bucket_dict.pop(nums[i-k] // (t + 1))  
            return False
    ```
    
3. **两个数组的交集 II**
    
    题目描述：给你两个整数数组 nums1 和 nums2，请你以数组形式返回两数组的交集。**返回结果中每个元素出现的次数，应与元素在两个数组中都出现的次数一致**（如果出现次数不一致，则考虑取较小值）。可以不考虑输出结果的顺序。
    
    ```python
    class Solution:
        def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
            hash_map = {}
            result = []
            # 统计nums1中各个元素的出现次数，hash_map[num]即为num在nums1中出现过的次数
            for num1 in nums1:
                if num1 in hash_map:
                    hash_map[num1] += 1
                else:
                    hash_map[num1] = 1
    
            for num2 in nums2:
                if num2 in hash_map and hash_map[num2] != 0:
    		        # 代码逻辑：如果num2在hash_map中，说明nums1中有num2；**hash_map[num2] -= 1**是为了**扣除一次nums2和nums1中num2重合的次数**，并同时在result中记录一次num2的出现；如果hash_map[num2]已经为0，说明nums1中num2的数量小于等于nums2中num2的数量，因为取较小数量，所以不能再执行重合次数的减少并在结果中添加num2
                    **hash_map[num2] -= 1**
                    result.append(num2)
            return result
    ```
    
4. 数独有效性
    
    请你判断一个 `9 x 9` 的数独是否有效。只需要 **根据以下规则** ，验证已经填入的数字是否有效即可。
    
    1. 数字 `1-9` 在每一行只能出现一次。
    2. 数字 `1-9` 在每一列只能出现一次。
    3. 数字 `1-9` 在每一个以粗实线分隔的 `3x3` 宫内只能出现一次。（请参考示例图）
    
    **注意：**
    
    - 一个有效的数独（部分已被填充）不一定是可解的。
    - 只需要根据以上规则，验证已经填入的数字是否有效即可。
    - 空白格用 `'.'` 表示。
    
    ```python
    class Solution(object):
        def isValidSudoku(self, board):
            """
            :type board: List[List[str]]
            :rtype: bool
            """
            # 大小为9的数组的每个元素为一个大小为10的数组，大小为10的数组下标为数独中真实数字，对应的数组值为该数字在该行/列/方格中的出现次数（9行/9列/9个方格）
            row = [[0 for _ in range(10)] for _ in range(9)]
            col = [[0 for _ in range(10)] for _ in range(9)]
            box = [[0 for _ in range(10)] for _ in range(9)]
            for i in range(9):
                for j in range(9):
                    if board[i][j] != '.':
                        num = int(board[i][j])
                        box_index = (i // 3) * 3 + j // 3
                        if row[i][num] + col[j][num] + box[box_index][num] > 0:
                            return False
                        row[i][num] = 1
                        col[j][num] = 1
                        box[box_index][num] = 1
            return True
    
    ```
    
5. **找不同**
    
    题目描述：给定两个字符串 s 和 t，它们只包含小写字母。字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。请找出在 t 中被添加的字母。
    
    🌟使用Counter类
    
    collections.`Counter`是`dict`的一个子类，用于计数可哈希对象。它是一个集合，其中元素存储为字典键，它们的计数存储为字典的值。
    
    https://blog.csdn.net/wei0514wei/article/details/143447517
    
    ```python
    from collections import Counter
     
    # 从可迭代对象创建
    c = Counter(['a', 'b', 'c', 'a', 'b', 'b'])
    print(c)  # Counter({'b': 3, 'a': 2, 'c': 1})
     
    # 从字符串创建
    c = Counter('abracadabra')
    print(c)  # Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})
     
    # 从字典创建
    c = Counter({'red': 4, 'blue': 2})
    print(c)  # Counter({'red': 4, 'blue': 2})
     
    # 使用关键字参数
    c = Counter(cats=4, dogs=8)
    print(c)  # Counter({'dogs': 8, 'cats': 4})
    
    # 访问和修改计数
    c = Counter('abracadabra')
     
    # 访问计数
    print(c['a'])  # 5
    print(c['z'])  # 0 (不存在的元素返回0，而不是引发KeyError)
     
    # 设置计数
    c['b'] = 3
    print(c)  # Counter({'a': 5, 'b': 3, 'r': 2, 'c': 1, 'd': 1})
     
    # 删除元素
    del c['b']
    print(c)  # Counter({'a': 5, 'r': 2, 'c': 1, 'd': 1})
    
    # 高级操作
    # 1. elements()：返回一个迭代器，其中每个元素重复计数次
    c = Counter(a=4, b=2, c=0, d=-2)
    print(list(c.elements()))  # ['a', 'a', 'a', 'a', 'b', 'b']
    
    # 2. most_common(n)：返回一个列表，其中包含n个最常见的元素及其计数，按计数从高到低排序
    c = Counter('abracadabra')
    print(c.most_common(3))  # [('a', 5), ('b', 2), ('r', 2)]
    
    # 3. subtract([iterable-or-mapping])：从迭代对象或另一个映射（或计数器）中减去元素。
    c = Counter(a=4, b=2, c=0, d=-2)
    d = Counter(a=1, b=2, c=3, d=4)
    c.subtract(d)
    print(c)  # Counter({'a': 3, 'b': 0, 'c': -3, 'd': -6})
    print(list(c - d)[0]) # 'a'
    # list(Counter)会提取Counter中的键组成列表而舍弃值
    ```
    

# 二叉树

🌟很好用的第三方模块**`sortedcontainers`**的使用教学【默认为升序】：

https://blog.csdn.net/Supreme7/article/details/132837013

完全二叉树的特点：**完全二叉树的左右子树也是完全二叉树**。**完全二叉树的左右子树中，至少有一棵是满二叉树**。

平衡二叉树的特点：N个节点的平衡二叉树的树高**一定**为O(logN)

## 例题

### BFS算法常用于寻找最短路径

**111. 二叉树的最小深度** | [**力扣**](https://leetcode.cn/problems/minimum-depth-of-binary-tree/) | [**LeetCode**](https://leetcode.com/problems/minimum-depth-of-binary-tree/) |  🟢

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

**示例 1：**

![](https://labuladong.online/algo/images/lc/uploads/2020/10/12/ex_depth.jpg)

```
输入：root = [3,9,20,null,null,15,7]
输出：2
```

DFS 递归遍历的解法：

```python
class Solution:
    def __init__(self):
        # 记录最小深度（根节点到最近的叶子节点的距离）
        self.minDepthValue = float('inf')
        # 记录当前遍历到的节点深度
        self.currentDepth = 0

    def minDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0

        # 从根节点开始 DFS 遍历
        self.traverse(root)
        return self.minDepthValue

    def traverse(self, root: TreeNode) -> None:
        if root is None:
            return

        # 前序位置进入节点时增加当前深度
        self.currentDepth += 1

        # 如果当前节点是叶子节点，更新最小深度
        if root.left is None and root.right is None:
            self.minDepthValue = min(self.minDepthValue, self.currentDepth)

        self.traverse(root.left)
        self.traverse(root.right)

        # 后序位置离开节点时减少当前深度
        self.currentDepth -= 1
```

BFS 层序遍历的解法：

```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        q = deque([root])
        # root 本身就是一层，depth 初始化为 1
        depth = 1

        while q:
            sz = len(q)
            # 遍历当前层的节点
            for _ in range(sz):
                cur = q.popleft()
                # 判断是否到达叶子结点
                if cur.left is None and cur.right is None:
                    return depth
                # 将下一层节点加入队列
                if cur.left is not None:
                    q.append(cur.left)
                if cur.right is not None:
                    q.append(cur.right)
            # 这里增加步数
            depth += 1
        return depth
```

https://blog.csdn.net/zqx951102/article/details/128208737

deque底层是双向链表，增删的时间复杂度都更低（O(1)）

### DFS算法常用于寻找全部路径

如果用BFS寻找全部路径，需要在原来的节点类型基础上包装一个新类State用来存储该节点的路径信息，eg

```jsx
// 自定义类，path保存节点以及根节点到该节点的路径
function State(node, path) {
    this.node = node;
    this.path = path;
}
```

## 二叉搜索树及TreeMap类（key用于简历二叉搜索树，value存储额外的值，形成键值对）

Java的TreeMap类对应python的`sortedcontainers.SortedDict`类，TreeSet对应`sortedcontainers.SortedSet`类

# 二叉堆

二叉堆的主要操作就两个，`sink`（下沉）和 `swim`（上浮），用以维护二叉堆的性质。

二叉堆的主要应用有两个，首先是一种很有用的数据结构优先级队列（Priority Queue），第二是一种排序方法堆排序（Heap Sort）。

二叉堆就是一种能够动态排序的数据结构，能动态排序的常用数据结构其实只有两个，一个是优先级队列（底层用二叉堆实现），另一个是二叉搜索树。

在Python中，实现优先队列【默认都是小顶堆】主要有几种方式，包括使用内置的`heapq`模块（import heapq）和第三方库如`queue.PriorityQueue`（from queue import PriorityQueue，`queue.PriorityQueue`实际上是线程安全的，如果你的应用场景是多线程的，可以考虑使用）。

- heapq用法
    
    ```python
    import heapq
     
    # 创建一个空列表作为堆
    heap = []
     
    # 添加元素到堆中，元素本身作为优先级
    heapq.heappush(heap, (priority, item))
     
    # 弹出堆顶元素（最小元素）
    item = heapq.heappop(heap)
     
    # 查看堆顶元素但不弹出
    item = heap[0]
     
    # 遍历所有元素（注意：这会破坏堆的结构）
    while heap:
        item = heapq.heappop(heap)
        print(item)
    ```
    
- queue.PriorityQueue用法
    
    ```python
    from queue import PriorityQueue
     
    # 创建一个优先队列实例
    pq = PriorityQueue()
     
    # 添加元素到队列中，元素的优先级越小，越先被处理（在Python中，较小的数字表示较高的优先级）
    pq.put((priority, item))
     
    # 获取队列中的元素（弹出并返回最小的元素）
    item = pq.get()
     
    # 遍历所有元素直到队列为空
    while not pq.empty():
        item = pq.get()
        print(item)
    ```
    

# 图

**寻找从节点n到节点m的所有路径**时一般使用**DFS算法**，如果未说明图是否是有向无环图的话，需要使用一个onPath数组记录某个节点是否在路径上，防止出现死循环；但是如果说明图是有向无环图的话，只需要直接使用path数组记录当前路径上有什么节点就行，不需要额外使用onPath数组。

- 代码示例
    
    使用onPath数组避免死循环
    
    ```python
    # 下面的算法代码可以遍历图的所有路径，寻找从 src 到 dest 的所有路径
    
    # onPath 和 path 记录当前递归路径上的节点
    **on_path = [False] * len(graph)**
    path = []
    
    def traverse(graph, src, dest):
        # base case
        if src < 0 or src >= len(graph):
            return
        **if on_path[src]:
            # 防止死循环（成环）
            return**
        # 前序位置
        on_path[src] = True
        path.append(src)
        if src == dest:
            print(f"find path: {path}")
        for e in graph.neighbors(src):
            traverse(graph, e.to, dest)
        # 后序位置
        path.pop()
        **on_path[src] = False**
    ```
    
    有向无环图不使用onPath数组
    
    ```python
    class Solution:
        # 记录所有路径
        def __init__(self):
            self.res = []
            self.path = []
    
        def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
            self.traverse(graph, 0)
            return self.res
    
        # 图的遍历框架
        def traverse(self, graph: List[List[int]], s: int):
            # 添加节点 s 到路径
            self.path.append(s)
    
            n = len(graph)
            if s == n - 1:
                # 到达终点
                self.res.append(self.path.copy())
                self.path.pop()
                return
    
            # 递归每个相邻节点
            for v in graph[s]:
                self.traverse(graph, v)
    
            # 从路径移出节点 s
            self.path.pop()
    ```
    

BFS 算法一般只用来寻找那条**最短路径**，不会用来求**所有路径**。如果只求最短路径的话，只需要遍历「节点」就可以了，因为**按照 BFS 算法一层一层向四周扩散的逻辑，第一次遇到目标节点，必然就是最短路径。**

在BFS遍历节点时，要在将节点append到队列时就将节点visited值设为true，而不是将节点pop出队列的时候才将visited数组中对应位置设置为true，因为这样可能会导致节点被重复添加到队列中！！

```python
# 图结构的 BFS 遍历，从节点 s 开始进行 BFS，且（在state类中的weight属性）记录路径的权重和
# 每个节点自行维护 State 类，记录从 s 走来的权重和
class State:
    def __init__(self, node, weight):
        # 当前节点 ID
        self.node = node
        # 从起点 s 到当前节点的权重和
        self.weight = weight

def bfs(graph, s):
    visited = [False] * len(graph)
    from collections import deque

    q = deque([State(s, 0)])
    visited[s] = True

    while q:
        state = q.popleft()
        cur = state.node
        weight = state.weight
        print(f"visit {cur} with path weight {weight}")
        for e in graph.neighbors(cur):
            **if not visited[e.to]:**
                q.append(State(e.to, weight + e.weight))
                **visited[e.to] = True**
```

【对于加权图，**由于每条边的权重不同，遍历的步数不再能代表最短路径的长度，所以需要每个节点用自定义State类维护自己的路径权重和**，最典型的例子就是[**Dijkstra 单源最短路径算法**](https://labuladong.online/algo/data-structure/dijkstra/)了。】

## dijkstra算法：寻找最短路径

- 代码实现
    
    ```python
    from typing import List
    from queue import PriorityQueue
    
    class State:
        def __init__(self, id, distFromStart):
            self.id = id
            # distFromStart属性记录从startNode节点到当前节点的距离
            self.distFromStart = distFromStart
    
    # 输入一幅图和一个起点 start，计算 start 到其他节点的最短距离
    def dijkstra(start:int, graph:Graph) -> List[int]:
        # 图中节点的个数
        V = len(graph)
        # 记录最短路径的权重，你可以理解为 dp table
        # 定义：distTo[i] 的值就是节点 start 到达节点 i 的最短路径权重
        distTo = [0] * V
        # 求最小值，所以 dp table 初始化为正无穷
        for i in range(V):
            distTo[i] = float('inf')
        # base case，start 到 start 的最短距离就是 0
        distTo[start] = 0
    
        # 优先级队列，distFromStart 较小的排在前面
        pq = PriorityQueue(lambda a,b: a.distFromStart - b.distFromStart)
    
        # 从起点 start 开始进行 BFS
        pq.put(State(start, 0))
    
        while not pq.empty():
        **#** **while循环中更新的对象永远不是startNode到当前节点的最短距离，而是startNode到当前节点的邻居节点的最短距离**
            curState = pq.get() 
            curNodeID = curState.id
            curDistFromStart = curState.distFromStart
    
            if curDistFromStart > distTo[curNodeID]:
                # 已经有一条更短的路径从 startNode 到达 curNode 节点了，**也就**说明已经有一条更短的经过curNode的从startNode到达curNode的所有邻居的路径了
                # 当前state类节点**curState**记录的**distFromStart**不一定是从start到当前节点的最短路径；已经有一条更短的路径到达 curNode 节点也说明之前更新distTo[curNodeID]时已经经历过下面的对curNodeID的相邻节点的最短路径检查更新，并且已经经过前序节点进行过对当前节点在distTo表中的更新，因此无需再次进行下面的for循环更新；**这段if语句就是确保不会进入死循环的关键**
                **continue**
           
            # 将 curNode 的相邻节点装入队列
            for nextNodeID in graph.neighbors(curNodeID):
                # 看看从 startNode 达到 nextNode 的距离是否会更短
                distToNextNode = distTo[curNodeID] + graph.weight(curNodeID, nextNodeID)
                if distTo[nextNodeID] > distToNextNode:
                    **# 更新 dp table，在这一步更新了distTo表，就不需要再在for循环之前执行代码 distTo[curNodeID] = curDistFromStart 来更新distTo表；State类型节点只会在这一步创建并加入队列，因此天然地 curState.distFromStart 一定与 distTo[curNodeID] 同步过
                    distTo[nextNodeID] = distToNextNode**
                    # 将这个节点以及距离放入队列
                    **pq.put(State(nextNodeID, distToNextNode))**
        return distTo
    ```
    
    只寻找startNode到指定endNode的最短路径
    
    ```python
    def dijkstra(start: int, end: int, graph: List[List[int]]) -> int:
        # ...
    
        while pq:
            curState = heapq.heappop(pq)
            curNodeID, curDistFromStart = curState.id, curState.distFromStart
    
            **# 在这里加一个判断就行了，其他代码不用改
            if curNodeID == end:
                return curDistFromStart**
    
            if curDistFromStart > distTo[curNodeID]:
                continue
    
            # ...
    
        # 如果运行到这里，说明从 start 无法走到 end
        return float('inf')
    ```
    
    因为优先级队列自动排序的性质，**每次**从队列里面拿出来的都是 `distFromStart` 值最小的，所以当你**第一次**从队列中拿出终点 `end` 时，此时的 `distFromStart` 对应的值就是从 `start` 到 `end` 的最短距离。这个算法较之前的实现提前 return 了，所以效率有一定的提高。
    

时间复杂度：

理想情况下优先级队列中最多装 `V` 个节点，对优先级队列的操作次数和 `E` 成正比，所以整体的时间复杂度就是 O(ElogV)。

但是本文实现的 Dijkstra 算法，使用了 Java 的 `PriorityQueue` 这个数据结构，这个容器类底层使用二叉堆实现，但没有提供通过索引操作队列中元素的 API，所以队列中会有重复的节点，最多可能有 `E` 个节点存在队列中。所以本文实现的 Dijkstra 算法复杂度并不是理想情况下的 O(ElogV)*O*(*ElogV*)，而是 O(ElogE)*O*(*ElogE*)，可能会略大一些，因为图中边的条数一般是大于节点的个数的。

# 做题技巧

## 矩阵

矩阵旋转：

顺时针旋转90度 == 按左上到右下的对角线镜像对称，再逐行反转
（右上角元素：↙️ + ➡️ = ⬇️）

逆时针旋转90度 == 按右上到左下的对角线镜像对称，再逐行反转
（左上角元素：↘️ + ⬅️ = ⬇️）

## 链表

🌟反转链表：**206. 反转链表：经典的递归算法**

🌟回文链表：234. 回文链表：判断当前链表是否为回文链表

方法1：用快慢指针找到中间节点之后反转中间节点（不含）之后的链表，再以原链表头为初始left，反转之后的后半段链表新头（也就是原链表尾）为初始right，左右指针同步向前，一直判断到right为None，如果都没有出现不一样的值就可以判断原链表为回文链表

🌟环形链表：141. 环形链表、142. 环形链表II

方法：用快慢指针，快指针每一次走两步，慢指针每一次走一步，如果**快慢指针同时非空并相遇**说明存在环；在相遇时将其中一个指针置为原链表头（另一个保留为上一次相遇的节点）并再次同步伐向前遍历，第一次相遇就是环起点

- 原理
    
    我们假设快慢指针相遇时，慢指针 `slow` 走了 `k` 步，那么快指针 `fast` 一定走了 `2k` 步：
    
    ![image.png](leedcode%E5%88%B7%E9%A2%98%20197e64a566218085a670d86607670095/image.png)
    
    `fast` 一定比 `slow` 多走了 `k` 步，这多走的 `k` 步其实就是 `fast` 指针在环里转圈圈，所以 `k` 的值就是环长度的「整数倍」。
    
    假设相遇点距环的起点的距离为 `m`，那么结合上图的 `slow` 指针，环的起点距头结点 `head` 的距离为 `k - m`，也就是说如果从 `head` 前进 `k - m` 步就能到达环起点。
    
    巧的是，如果从相遇点继续前进 `k - m` 步，也恰好到达环起点。因为结合上图的 `fast` 指针，从相遇点开始走k步可以转回到相遇点，那走 `k - m` 步肯定就走到环起点了：
    
    ![image.png](leedcode%E5%88%B7%E9%A2%98%20197e64a566218085a670d86607670095/image%201.png)
    
    所以，只要我们把快慢指针中的任一个重新指向 `head`，然后两个指针同速前进，`k - m` 步后一定会相遇，相遇之处就是环的起点了。
    

## 排序算法

🌟归并排序

时间复杂度：O(NlogN)，空间复杂度：O(N)（递归栈为树高，但是需要新建一个和原数组等长的新数组放置归并后的结果）

[**493. 翻转对**](https://labuladong.online/algo/practice-in-action/merge-sort/#_493-%E7%BF%BB%E8%BD%AC%E5%AF%B9)

[**315. 计算右侧小于当前元素的个数**](https://labuladong.online/algo/practice-in-action/merge-sort/#_315-%E8%AE%A1%E7%AE%97%E5%8F%B3%E4%BE%A7%E5%B0%8F%E4%BA%8E%E5%BD%93%E5%89%8D%E5%85%83%E7%B4%A0%E7%9A%84%E4%B8%AA%E6%95%B0)

🌟快速排序

- 代码实现
    
    ```python
    import random
    
    class Quick:
        @staticmethod
        def sort(nums: List[int]):
            # 为了避免出现耗时的极端情况，先随机打乱
            random.shuffle(nums)
            # 排序整个数组（原地修改）
            Quick.sort_(nums, 0, len(nums) - 1)
    
        @staticmethod
        def sort_(nums: List[int], lo: int, hi: int):
            if lo >= hi:
                return
            # 对 nums[lo..hi] 进行切分
            # 使得 nums[lo..p-1] <= nums[p] < nums[p+1..hi]
            p = Quick.partition(nums, lo, hi)
    
            Quick.sort_(nums, lo, p - 1)
            Quick.sort_(nums, p + 1, hi)
        
        # 对 nums[lo..hi] 进行切分
        @staticmethod
        def partition(nums: List[int], lo: int, hi: int) -> int:
            pivot = nums[lo]
            # 关于区间的边界控制需格外小心，稍有不慎就会出错
            # 我这里把 i, j 定义为开区间，同时定义：
            # [lo, i) <= pivot；(j, hi] > pivot
            # 之后都要正确维护这个边界区间的定义
            i, j = lo + 1, hi
            # 当 i > j 时结束循环，以保证区间 [lo, hi] 都被覆盖
            while i <= j:
                while i < hi and nums[i] <= pivot:
                    i += 1
                    # 此 while 结束时恰好 nums[i] > pivot
                while j > lo and nums[j] > pivot:
                    j -= 1
                    # 此 while 结束时恰好 nums[j] <= pivot
    
                if i >= j:
    		            # i == j的时候没必要再进行一次交换，可以直接break出去；i > j的时候说明当前i指向的元素大于pivot且j指向的元素小于等于pivot，不需要进行交换
                    break
                # 此时 [lo, i) <= pivot && (j, hi] > pivot，交换 nums[j] 和 nums[i]
                nums[i], nums[j] = nums[j], nums[i]
                # 此时 [lo, i] <= pivot && [j, hi] > pivot
            # 此时j指向的元素一定是小于等于pivot的，而pivot的原位置在最左边，因此是把pivot换到j处而不是换到i处；最后将 pivot 放到合适的位置，即 pivot 左边元素较小，右边元素较大
            nums[lo], nums[j] = nums[j], nums[lo]
            return j
    ```
    
- 复杂度分析
    
    显然，快速排序的时间复杂度主要消耗在 `partition` 函数上，因为这个函数中存在循环。
    
    所以 `partition` 函数到底执行了多少次？每次执行的时间复杂度是多少？总的时间复杂度是多少？
    
    和归并排序类似，需要结合之前画的这幅图来从整体上分析：
    
    ![](https://labuladong.online/algo/images/quick-sort/4.jpeg)
    
    **`partition` 执行的次数是二叉树节点的个数，每次执行的复杂度就是每个节点代表的子数组 `nums[lo..hi]` 的长度，所以总的时间复杂度就是整棵树中「数组元素」的个数**。
    
    假设数组元素个数为 `N`，那么二叉树每一层的元素个数之和就是 O(N)；切分点 `p` 每次都落在数组正中间的理想情况下，树的层数为 O(logN)，所以理想的总时间复杂度为 **O(NlogN)**。
    
    由于快速排序没有使用任何辅助数组，所以空间复杂度就是递归堆栈的深度，也就是**树高 O(logN)**。
    
    当然，我们之前说过快速排序的效率存在一定随机性，如果每次 `partition` 切分的结果都极不均匀：
    
    ![](https://labuladong.online/algo/images/quick-sort/3.jpeg)
    
    快速排序就退化成选择排序了，树高为 O(N)，每层节点的元素个数从 `N` 开始递减，总的时间复杂度为：
    
    ```
    N + (N - 1) + (N - 2) + ... + 1 = O(N^2)
    ```
    
    所以我们说，**快速排序理想情况的时间复杂度是 O(NlogN)，空间复杂度 O(logN)，极端情况下的最坏时间复杂度是 O(N2)，空间复杂度是 O(N)。**
    
    基于快速排序进行快速选择的时候，时间复杂度是O(N)：
    
    例题：https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?envType=study-plan-v2&envId=top-100-liked
    

## 二分查找

双闭区间

左边界

```python
def left_bound(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
		    mid = left + (right - left) // 2
		    if nums[mid] == taregt:
		        right = mid - 1
		    elif nums[mid] < target:
		        left = mid + 1
		    elif nums[mid] > target:
			      right = mid - 1
		# left为小于target的最大索引或第一个target值的索引（nums非降序）
		return left
```

右边界

```python
def right_bound(nums, target):
		left, right = 0, len(nums) - 1
		while left <= right:
				mid = left + (right - left) // 2
				if nums[mid] == target:
						left = mid + 1
				elif nums[mid] < target:
		        left = mid + 1
		    elif nums[mid] > target:
			      right = mid - 1
		# right = left - 1 为大于target的最小索引或最后一个target值的索引（nums非降序）
		# 由于while的结束条件一定是left == right + 1，所以left - 1一定为right
		return right
```

O(log(m+n))时间复杂度二分查找两个升序数组合并后的中位数：https://leetcode.cn/problems/median-of-two-sorted-arrays/description/?envType=study-plan-v2&envId=top-100-liked

## 栈

维护一个可以在常数时间获得栈中最小值的栈

https://leetcode.cn/problems/min-stack/?envType=study-plan-v2&envId=top-100-liked

思路：在维护主栈的同时维护一个当前值入栈时的最小值栈，最小值就是最小值栈的最后一个元素；弹出栈顶元素时需要检查当前栈顶元素是否为最小值，若是，需要看弹出栈顶元素后的新栈顶元素与最小值栈新栈顶元素比大小，更新最小值栈

对顶堆：用于查找数据流的中位数

## 贪心&动规

股票问题：https://labuladong.online/algo/dynamic-programming/stock-problem-summary/#_121-%E4%B9%B0%E5%8D%96%E8%82%A1%E7%A5%A8%E7%9A%84%E6%9C%80%E4%BD%B3%E6%97%B6%E6%9C%BA

核心是建立两个状态转移方程：dp[i][k][0]和dp[i][k][1]，  注意边界情况分析