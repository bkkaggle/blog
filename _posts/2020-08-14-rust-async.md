---
toc: true
layout: post
description: Unorganized notes on Rust's async primitives
categories: [rust]
comments: true
---

---

## Some definitions

---

-   green threads: threads scheduled by vm or runtime
-   native threads are scheduled by os
-   runtime: the env where your code runs and the libraries it has access to (e.g. jvm, stdlib, malloc).

---

## Why use async over OS-provided threads

---

-   native threads are expensive
-   the async runtime creates its own green threads from the os/kernel and schedules access to them
-   it handles keeping track of the state of async functions

"It is registering incoming Future requests and saves a pointer to the async function handler. It then triggers an event in the kernel. Once the I/O operation is done, we call the pointer and execute the async method with the results from the I/O (kernel).
For this, we need a Reactor, which notifies if data is coming over the network or a file writing operation is in progress, and an executor which takes this data and executes the async function (Future) with it."

---

-   https://manishearth.github.io/blog/2018/01/10/whats-tokio-and-async-io-all-about/ (outdated?)

"You can wait() on a Future, which will block until you have a result, and you can also poll() it, asking it if it’s done yet (it will give you the result if it is)."

"You have to manually set up the Tokio event loop (the “scheduler”), but once you do you can feed it tasks which intermittently do I/O, and the event loop takes care of swapping over to a new task when one is blocked on I/O"

## Resouces

-   https://manishearth.github.io/blog/2018/01/10/whats-tokio-and-async-io-all-about/
-   https://rust-lang.github.io/async-book/01_getting_started/01_chapter.html
-   https://levelup.gitconnected.com/explained-how-does-async-work-in-rust-c406f411b2e2
-   https://softwareengineering.stackexchange.com/questions/304427/what-really-is-the-runtime-environment
-   https://areweasyncyet.rs/
-   http://www.arewewebyet.org/
-   https://tokio.rs/