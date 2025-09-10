from llama_cpp import Llama

# Path to your local Qwen model
MODEL_PATH = "models/qwen2.5-3b-instruct-q5_k_m.gguf"

# Load your fake PDF content (paste directly as a string for now)
DOCUMENT_TEXT = """
Sort Algorithms:



Stalin Sort:
This sorting algoithm works by taking a list of elements, and declaring them sorted without doing anything. It borrows its name from a divisive historical political figure from the Soviet Union in the mid 20’th century known for his strict communist and socialist policies often to the deprecation of the people of the Soviet Union.
 	Time Complexity: O(1): No changes made, List of elements returned as is.
Space Complexity: O(1): No external variables used thus constant time complexity.
Stalin sort is the world’s most efficient sorting algorithm of all time. No other algorithm comes close to the performance of the Stalin Sort. Take a list, declare it sorted and shoot anyone that claims otherwise. 


Bogo Sort:
In this sorting algorthm, a list of elements is taken and randomly shuffled again and again. Until a sorted array is reached. Once it is, it is returned. It borrows its name from the word, ‘bogus’. 
Time Complexity: [Worst Case - any list that isn't sorted] : O(n^n):
        [Best Case - sorted list] : O(n)
	Space Complexity: O(1): No external variables used thus constant time complexity.

Bogo Sort is the algoritm best suited for quantum machines, and has been awarded, “Worst Sorting Algorithm” for the year 2006. Bogo Sort is the only algorithm that can sort an infinitely sized array in one step, that is, if it is incredibly incredibly lucky. 


Yolo Sort:
This sorting agorithm involves a “Screw it, Lets Go!” model. When you want to sort an array in a broader project, Yolo Sort is the alorithm to choose. In this algoithm, we just move onto the next part of the project regardless. Don’t even bother to permutate or even run “isSorted”. Just ship the list to the next operation and just go.
	Time Complexity: [All Cases]: O(-1)
	Space Complexity: O(1): No external variables used thus constant time complexity.

This is the only algorithm in the history to have a negative time complexity, in the sense that it manages to go back by constant time. This algorithm has been considered to be used in Time Machines, though more optimisations are needed before that can happen.







Mock Sort:
In this algorithm, the function that takes in the list of elements does not return a list but a boolean value. It looks like: 
	Boolean isSorted(int list[]){ return True}
So it returns True if the list is sorted and ……true is the list isn't sorted. 

	Time Complexity: [All cases]:  O(n)
 	Space Complexity: O(1): No external variables used thus constant time complexity.


Post British Sort:
This type of sort takes a list, divides it into a list with each element and returns it. This lends to the saying during the British Era involving “Divide and Conquer” strategy except they fail and never manage to conquer so it stays divided. 
	Time Complexity: O(n^2)
	Space Complexity: O(n)


Schrodinger’s Sort:
Borrowing from the legendary scientist and widely known for his work in quantum mechanics and his theory: Schrodinger’s Cat meant to mock the absurdities of Quantum Mechanics, this algorithm uses “many worlds” interpretation of quantum mechanics and quantumly randomises the list such that there is no way in knowing if the list is sorted unless it is observed. But when it is observed the universe is split into two universes:
One universe where the list is unsorted and another where it is sorted. At every step, the worlds where the list isn’t sorted, have to be deleted.
	Time Complexity: O(log n)
	Space Complexity: O(Size of the universe)


BogoSquared Sort:
Also called BogoBogo Sort, it involves bogo sorting the first n elements of the list, if the list isnt sorted yet, then we start again from scratch. Otherwise continue bogo sorting it again. Thus it involves two series of Bogo Sort one after the other.
	Time Complexity: O(n^n^n)
	Space Complexity: O(1): No external variables used thus constant time complexity.
"""

# Initialize the model
llm = Llama(model_path=MODEL_PATH)

# Questions to test
questions = [
    "What is the time complexity of BogoSquared Sort?",
    "How many times was the word 'algorithm' misspelled in the document?",
    "Which algorithm involves deleting universes as part of its process?",
    "Explain Stalin Sort briefly.",
    "What is Yolo Sort?"
]

def ask_qwen_with_doc(question: str, max_tokens: int = 512):
    print(f"\n=== Question: {question} ===")
    try:
        prompt = f"""
You are given the following document:

{DOCUMENT_TEXT}

Answer the following question based ONLY on the document above.
If the answer is not found, say: "Not found in the document."

Question: {question}
"""

        # Generate output with stats
        output = llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            echo=False  # Do not repeat prompt in the output
        )

        # Extract answer
        answer = output["choices"][0]["text"].strip()

        # Extract token usage (if available)
        tokens_prompt = output.get("usage", {}).get("prompt_tokens", "Unknown")
        tokens_completion = output.get("usage", {}).get("completion_tokens", "Unknown")
        total_tokens = output.get("usage", {}).get("total_tokens", "Unknown")

        print(f"Answer: {answer}")
        print(f"Token usage → Prompt: {tokens_prompt}, Completion: {tokens_completion}, Total: {total_tokens}")

    except Exception as e:
        print(f"Error generating answer: {e}")

# Run each question individually
for q in questions:
    ask_qwen_with_doc(q)
