public class Solution{public IList<string> FindWords(char[][] board, string[] words){List<string> res = new List<string>();TrieNode root = BuildTrie(words);for(int i = 0;i < board.Length;i++){for(int j = 0;j < board[0].Length;j++){Dfs(board, i, j, root, res);}}return res;}public void Dfs(char[][] board, int i, int j, TrieNode p, List<string> res){char c = board[i][j];if(c == '#' || p.Next[c - 'a'] == null) return;p = p.Next[c - 'a'];if(p.Word != null){res.Add(p.Word);p.Word = null;}board[i][j] = '#';if(i > 0) Dfs(board, i - 1, j, p, res);if(j > 0) Dfs(board, i, j - 1, p, res);if(i < board.Length - 1) Dfs(board, i + 1, j, p, res);if(j < board[0].Length - 1) Dfs(board, i, j + 1, p, res);board[i][j] = c;}public TrieNode BuildTrie(string[] words){TrieNode root = new TrieNode();foreach(string w in words){TrieNode p = root;foreach(char c in w){int i = c - 'a';if(p.Next[i] == null) p.Next[i] = new TrieNode();p = p.Next[i];}p.Word = w;}return root;}}public class TrieNode{public TrieNode[] Next = new TrieNode[26];public string Word;}
public class WordDictionary{private WordDictionary[] children;private bool isEndOfWord;public WordDictionary(){children = new WordDictionary[26];isEndOfWord = false;}public void AddWord(string word){WordDictionary curr = this;foreach(char c in word){if(curr.children[c - 'a'] == null) curr.children[c - 'a'] = new WordDictionary();curr = curr.children[c - 'a'];}curr.isEndOfWord = true;}public bool Search(string word){return Search(word, 0, this);}private bool Search(string word, int index, WordDictionary node){if(node == null) return false;if(index == word.Length) return node.isEndOfWord;char c = word[index];if(c == '.'){foreach(WordDictionary child in node.children){if(child != null && Search(word, index + 1, child)) return true;}return false;}else{return Search(word, index + 1, node.children[c - 'a']);}}}
public class Solution{public IList<IList<int>> ZigzagLevelOrder(TreeNode root){IList<IList<int>> sol = new List<IList<int>>();Travel(root, sol, 0);return sol;}private void Travel(TreeNode curr, IList<IList<int>> sol, int level){if(curr == null) return;if(sol.Count <= level){IList<int> newLevel = new List<int>();sol.Add(newLevel);}IList<int> collection = sol[level];if(level % 2 == 0) collection.Add(curr.val);else collection.Insert(0, curr.val);Travel(curr.left, sol, level + 1);Travel(curr.right, sol, level + 1);}}
public class Solution{public bool IsValidBST(TreeNode root){return IsValidBST(root, long.MinValue, long.MaxValue);}public bool IsValidBST(TreeNode root, long minVal, long maxVal){if(root == null) return true;if(root.val >= maxVal || root.val <= minVal) return false;return IsValidBST(root.left, minVal, root.val) && IsValidBST(root.right, root.val, maxVal);}}
public class Solution{int minDiff = int.MaxValue;TreeNode prev;public int GetMinimumDifference(TreeNode root){Inorder(root);return minDiff;}public void Inorder(TreeNode root){if(root == null) return;Inorder(root.left);if(prev != null) minDiff = Math.Min(minDiff, root.val - prev.val);prev = root;Inorder(root.right);}}
public class Solution{private Dictionary<int, Node> map = new Dictionary<int, Node>();public Node CloneGraph(Node node){return Clone(node);}private Node Clone(Node node){if(node == null) return null;if(map.ContainsKey(node.val)) return map[node.val];Node newNode = new Node(node.val, new List<Node>());map.Add(newNode.val, newNode);foreach(Node neighbor in node.neighbors) newNode.neighbors.Add(Clone(neighbor));return newNode;}}
public int SnakesAndLadders(int[][] board){int n = board.Length;Queue<int> queue = new Queue<int>();queue.Enqueue(1);bool[] visited = new bool[n * n + 1];for(int move = 0;!queue.Count.Equals(0);move++){for(int size = queue.Count;size > 0;size--){int num = queue.Dequeue();if(visited[num]) continue;visited[num] = true;if(num == n * n) return move;for(int i = 1;i <= 6 && num + i <= n * n;i++){int next = num + i;int value = GetBoardValue(board, next);if(value > 0) next = value;if(!visited[next]) queue.Enqueue(next);}}}return -1;}private int GetBoardValue(int[][] board, int num){int n = board.Length;int r =(num - 1) / n;int x = n - 1 - r;int y = r % 2 == 0 ? num - 1 - r * n : n + r * n - num;return board[x][y];}
public class LRUCache{class DLinkedNode{public int key;public int value;public DLinkedNode prev;public DLinkedNode next;}private void AddNode(DLinkedNode node){node.prev = head;node.next = head.next;head.next.prev = node;head.next = node;}private void RemoveNode(DLinkedNode node){DLinkedNode prev = node.prev;DLinkedNode next = node.next;prev.next = next;next.prev = prev;}private void MoveToHead(DLinkedNode node){RemoveNode(node);AddNode(node);}private DLinkedNode PopTail(){DLinkedNode res = tail.prev;RemoveNode(res);return res;}private Dictionary<int, DLinkedNode> cache = new Dictionary<int, DLinkedNode>();private int count;private int capacity;private DLinkedNode head, tail;public LRUCache(int capacity){this.count = 0;this.capacity = capacity;head = new DLinkedNode();head.prev = null;tail = new DLinkedNode();tail.next = null;head.next = tail;tail.prev = head;}public int Get(int key){if(!cache.ContainsKey(key)){return -1;}DLinkedNode node = cache[key];MoveToHead(node);return node.value;}public void Put(int key, int value){if(!cache.ContainsKey(key)){DLinkedNode newNode = new DLinkedNode();newNode.key = key;newNode.value = value;cache[key] = newNode;AddNode(newNode);++count;if(count > capacity){DLinkedNode tail = PopTail();cache.Remove(tail.key);--count;}}else{DLinkedNode node = cache[key];node.value = value;MoveToHead(node);}}}
public class Solution{public bool RootToLeafPathSum(TreeNode root, int targetSum, int sum){if(root == null) return false;if(root.left == null && root.right == null){sum += root.val;if(sum == targetSum) return true;}return RootToLeafPathSum(root.left, targetSum, sum + root.val) || RootToLeafPathSum(root.right, targetSum, sum + root.val);}public bool HasPathSum(TreeNode root, int targetSum){int sum = 0;return RootToLeafPathSum(root, targetSum, sum);}}