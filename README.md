# 🧠 C Programming & Data Structures

Welcome to the **C Programming and Data Structures** repository!

This repo is a comprehensive collection of **C programs**, **algorithms**, and **data structure implementations** — designed for beginners, students, and professionals revisiting core computer science concepts.

---

## 🚀 Features

- ✅ Well-structured C programs
- 📚 Classic data structure implementations
- 🧩 Algorithms with explanations
- 🧪 Example inputs/outputs for clarity
- 💡 Beginner-friendly and modular code
- ⚙️ Ready-to-compile source files

---

## 🧰 Tech Stack

| Component       | Description                          |
| --------------- | ------------------------------------ |
| **Language**    | C (C99/C11 Standard)                 |
| **Compiler**    | GCC / Clang                          |
| **IDE Support** | VS Code, Code::Blocks, Dev-C++, etc. |
| **Build Tools** | `make` (optional)                    |

---

## 📂 Repository Structure

```
C-Programming-and-Data-Structures/
├── README.md
├── 01_Basics/
│   ├── hello_world.c
│   ├── operators.c
│   └── control_statements.c
├── 02_Functions/
│   ├── recursion.c
│   └── function_pointers.c
├── 03_Arrays/
│   ├── linear_search.c
│   ├── binary_search.c
│   └── sorting_algorithms/
│       ├── bubble_sort.c
│       ├── selection_sort.c
│       └── quick_sort.c
├── 04_Strings/
│   ├── string_reverse.c
│   └── palindrome_check.c
├── 05_Data_Structures/
│   ├── linked_list.c
│   ├── stack.c
│   ├── queue.c
│   ├── binary_tree.c
│   └── graph.c
└── 06_Extras/
    ├── file_handling.c
    ├── macros.c
    └── dynamic_memory.c
```

---

## 📘 Topics Covered

<details>
<summary>🔹 C Fundamentals</summary>

- Variables, Data Types, Operators
- Control Statements (if-else, loops, switch)
- Functions, Recursion
- Pointers and Memory Management

</details>

<details>
<summary>🔹 Data Structures</summary>

- Arrays & Strings
- Linked Lists (Singly, Doubly, Circular)
- Stacks & Queues
- Trees & Binary Search Trees
- Graphs (Adjacency List/Matrix)
- Hash Tables

</details>

<details>
<summary>🔹 Algorithms</summary>

- Searching (Linear, Binary)
- Sorting (Bubble, Selection, Insertion, Merge, Quick)
- Traversal Algorithms (DFS, BFS)

</details>

---

## ⚡ Quick Start

```
# Clone the repository
git clone https://github.com/SatinderSinghSall/Programming-and-Data-Structures.git

# Navigate to the project folder
cd C-Programming-and-Data-Structures

# Compile and run an example
gcc 05_Data_Structures/linked_list.c -o linked_list
./linked_list
```

---

## 🧪 Example Program:

**`linked_list.c`**

```
#include <stdio.h>
#include <stdlib.h>

struct Node {
    int data;
    struct Node* next;
};

void printList(struct Node* n) {
    while (n != NULL) {
        printf("%d -> ", n->data);
        n = n->next;
    }
    printf("NULL\n");
}

int main() {
    struct Node* head = malloc(sizeof(struct Node));
    struct Node* second = malloc(sizeof(struct Node));
    struct Node* third = malloc(sizeof(struct Node));

    head->data = 1; head->next = second;
    second->data = 2; second->next = third;
    third->data = 3; third->next = NULL;

    printList(head);
    return 0;
}
```

**Output:**

```
1 -> 2 -> 3 -> NULL
```

---

## 🧑‍💻 Author

**Satinder Singh Sall**
💼 Full-Stack Developer | Programmer | Code Enthusiast
🔗 [Satinder Portfolio](https://satinder-portfolio.vercel.app/)
🔗 [GitHub Profile](https://github.com/SatinderSinghSall)
📧 [satindersinghsall111@gmail.com](satindersinghsall111@gmail.com)

---

## ⭐ Contributing

Contributions are always welcome!
To contribute:

1. Fork the repository 🍴
2. Create a new branch (`feature/your-feature`)
3. Commit your changes
4. Submit a Pull Request 🚀

---

> 💡 "First, solve the problem. Then, write the code." — John Johnson
