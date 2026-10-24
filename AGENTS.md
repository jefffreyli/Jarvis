Coding Principles
- Safe from bugs. Correctness (correct behavior right now) and defensiveness (correct behavior in the future) are required in any software we build.
- Easy to understand. The code has to communicate to future programmers who need to understand it and make changes in it (fixing bugs or adding new features). That future programmer might be you, months or years from now. You’ll be surprised how much you forget if you don’t write it down, and how much it helps your own future self to have a good design.
- Ready for change. Software always changes. Some designs make it easy to make changes; others require throwing away and rewriting a lot of code.
- Less is more, be minimal.
- Ensure modularity through abstraction and classes

Style
- Comments should only be added out of necessity, not explaining every little change. Be concise on a high level.
- Top level file doc string should be max one paragraph of text at the top explaining what the file does and the flow.
- There should not be long filler comments that are used as dividers. Instead of "#=====...", just have the title of the section "# Section header comment"
- Write short docstring for each new function. It should contain a short overview of what it does, then have a section for params, then return.
- Avoid using em dashes in comments
- Prioritize writing code that is easy to read and understand, so stay away from fancy short hands or writing everything all in one line.