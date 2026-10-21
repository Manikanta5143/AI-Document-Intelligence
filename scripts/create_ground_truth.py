import json
from pathlib import Path

ground_truth = {
    "metadata": {
        "evaluation_title": "Comprehensive RAG Evaluation and Hyperparameter Optimization",
        "document_name": "Manikanta_Ausula_Resume.pdf",
        "document_id": 3,
        "document_type": "Policy",
        "total_pages": 1,
        "total_chunks": 5,
        "chunk_ids": [11, 12, 13, 14, 15],
        "total_questions": 15,
        "categories": {
            "A": "Narrow factual questions",
            "B": "Project architecture questions",
            "C": "Multi-part questions",
            "D": "Questions requiring information from multiple resume sections",
            "E": "Unanswerable or partially answerable questions (abstention)"
        }
    },
    "chunks": [
        {
            "chunk_id": 11,
            "chunk_index": 0,
            "page_number": 1,
            "section_summary": "Header, Contact info, Professional Summary (B.Tech Class of 2027, CGPA 8.88/10, RAG Q&A system with Bedrock/S3, JWT, OOP, SOLID, 600+ DSA), Programming Languages (Python, Java, C++), Design Patterns (Singleton, Factory, Builder, Strategy, Observer)"
        },
        {
            "chunk_id": 12,
            "chunk_index": 1,
            "page_number": 1,
            "section_summary": "Technical Skills: Backend & APIs (FastAPI, Node.js, Express.js, RESTful APIs, JWT, WebSockets, Socket.IO), Frontend (React.js, React Router, Axios, HTML, CSS), Databases (MySQL, MongoDB, Snowflake, ChromaDB), AI/ML (LLMs, RAG, Embeddings, Transformers, Bedrock), Cloud & Tools (AWS S3, Git, GitHub, VS Code). Projects: Insurance AI Assistant (overview, FastAPI, S3, PyMuPDF, Titan Embeddings V2, ChromaDB)"
        },
        {
            "chunk_id": 13,
            "chunk_index": 2,
            "page_number": 1,
            "section_summary": "Projects: Insurance AI Assistant (Bedrock Nova 2 Lite zero-temperature generation, MySQL for metadata/Q&A, Snowflake for analytics). Zcoder (React.js, Node.js, Express.js, MongoDB, Socket.IO, 100+ coding problems, modular REST APIs, Mongoose, stateless JWT, bcrypt, Nodemailer with UUID tokens)"
        },
        {
            "chunk_id": 14,
            "chunk_index": 3,
            "page_number": 1,
            "section_summary": "Projects: Zcoder (real-time chat with Socket.IO). Dolphin: Student Opportunity Portal (React.js, Node.js, Express.js, MongoDB, 50+ hackathons/internships, RESTful APIs with JWT and bcrypt, responsive React.js frontend with React Router/Axios, bookmarking). Education: Jawaharlal Nehru Technological University, Hyderabad | 2023 - 2027"
        },
        {
            "chunk_id": 15,
            "chunk_index": 4,
            "page_number": 1,
            "section_summary": "Education: Jawaharlal Nehru Technological University, Hyderabad | 2023 - 2027 | B.Tech, Computer Science and Engineering (Artificial Intelligence & Machine Learning) | CGPA: 8.88/10. Achievements: 600+ DSA problems (LeetCode 300+, GeeksforGeeks 300+), Walmart Converge Hackathon Certificate of Appreciation"
        }
    ],
    "questions": [
        {
            "question_id": "Q01",
            "category": "Category A: Narrow factual",
            "category_code": "A",
            "question": "Which embedding model does my Insurance AI Assistant use for document and query embeddings?",
            "is_answerable": True,
            "relevant_chunk_ids": [12, 13],
            "reference_answer": "The Insurance AI Assistant uses Amazon Titan Text Embeddings V2 for document and query embeddings.",
            "expected_facts": [
                "The Insurance AI Assistant uses Amazon Titan Text Embeddings V2 for document and query embeddings."
            ]
        },
        {
            "question_id": "Q02",
            "category": "Category A: Narrow factual",
            "category_code": "A",
            "question": "What is my cumulative grade point average (CGPA) and degree program at Jawaharlal Nehru Technological University, Hyderabad?",
            "is_answerable": True,
            "relevant_chunk_ids": [11, 15],
            "reference_answer": "The candidate is pursuing a B.Tech in Computer Science and Engineering (Artificial Intelligence & Machine Learning), Class of 2027, at Jawaharlal Nehru Technological University, Hyderabad, with a CGPA of 8.88/10.",
            "expected_facts": [
                "Cumulative grade point average (CGPA) is 8.88/10.",
                "Degree is B.Tech in Computer Science and Engineering (Artificial Intelligence & Machine Learning), Class of 2027, at Jawaharlal Nehru Technological University, Hyderabad."
            ]
        },
        {
            "question_id": "Q03",
            "category": "Category A: Narrow factual",
            "category_code": "A",
            "question": "Which database stores the vector embeddings in the Insurance AI Assistant, and what framework powers its backend REST API?",
            "is_answerable": True,
            "relevant_chunk_ids": [12, 13],
            "reference_answer": "ChromaDB stores the vector embeddings, and FastAPI (in Python) powers the backend REST API.",
            "expected_facts": [
                "ChromaDB stores the vector embeddings.",
                "FastAPI (in Python) powers the backend REST API."
            ]
        },
        {
            "question_id": "Q04",
            "category": "Category A: Narrow factual",
            "category_code": "A",
            "question": "What programming languages and design patterns are listed under my technical skills?",
            "is_answerable": True,
            "relevant_chunk_ids": [11, 12],
            "reference_answer": "The programming languages listed are Python, Java, and C++. The design patterns listed are Singleton, Factory, Builder, Strategy, and Observer.",
            "expected_facts": [
                "Programming languages listed are Python, Java, and C++.",
                "Design patterns listed are Singleton, Factory, Builder, Strategy, and Observer."
            ]
        },
        {
            "question_id": "Q05",
            "category": "Category B: Project architecture",
            "category_code": "B",
            "question": "Explain the end-to-end RAG pipeline in my Insurance AI Assistant from document upload to answer generation.",
            "is_answerable": True,
            "relevant_chunk_ids": [12, 13],
            "reference_answer": "The Insurance AI Assistant RAG pipeline ingests policy documents stored in Amazon S3, extracts text using PyMuPDF, generates vector embeddings using Amazon Titan Text Embeddings V2, stores and retrieves chunks via similarity search in ChromaDB, and passes retrieved context to Amazon Bedrock Nova 2 Lite with zero temperature to generate grounded answers and summaries. Document metadata and Q&A history are logged in MySQL, and analytics data is stored in Snowflake.",
            "expected_facts": [
                "Policy documents are uploaded and stored in Amazon S3.",
                "PyMuPDF is used for PDF document extraction.",
                "Amazon Titan Text Embeddings V2 generates document and query embeddings.",
                "ChromaDB is used for vector storage and similarity search.",
                "Amazon Bedrock Nova 2 Lite generates context-grounded answers and summaries from retrieved policy content.",
                "MySQL stores document metadata and question-answer history, while Snowflake stores and analyzes application analytics data."
            ]
        },
        {
            "question_id": "Q06",
            "category": "Category B: Project architecture",
            "category_code": "B",
            "question": "What are the respective purposes of Amazon S3, MySQL, and Snowflake in my Insurance AI Assistant project?",
            "is_answerable": True,
            "relevant_chunk_ids": [12, 13],
            "reference_answer": "Amazon S3 stores raw insurance policy documents, MySQL stores document metadata and question-answer history, and Snowflake stores and analyzes application analytics data.",
            "expected_facts": [
                "Amazon S3 is used to store uploaded insurance policy documents.",
                "MySQL is used for storing document metadata and question-answer history.",
                "Snowflake is used to store and analyze application analytics data."
            ]
        },
        {
            "question_id": "Q07",
            "category": "Category B: Project architecture",
            "category_code": "B",
            "question": "How do Amazon Titan Text Embeddings V2, ChromaDB, and Amazon Bedrock Nova 2 Lite work together in the Insurance AI Assistant?",
            "is_answerable": True,
            "relevant_chunk_ids": [12, 13],
            "reference_answer": "Amazon Titan Text Embeddings V2 generates vector embeddings for documents and queries; ChromaDB stores these embeddings and performs similarity search to retrieve relevant chunks; and Amazon Bedrock Nova 2 Lite generates context-grounded answers from the retrieved chunks.",
            "expected_facts": [
                "Amazon Titan Text Embeddings V2 generates embeddings for document chunks and user queries.",
                "ChromaDB stores the vector embeddings and performs similarity search to retrieve the most relevant chunks.",
                "Amazon Bedrock Nova 2 Lite receives retrieved policy context to generate context-grounded answers and summaries."
            ]
        },
        {
            "question_id": "Q08",
            "category": "Category C: Multi-part",
            "category_code": "C",
            "question": "Describe Zcoder's technology stack, authentication mechanisms, code execution, and real-time chat functionality.",
            "is_answerable": True,
            "relevant_chunk_ids": [13, 14],
            "reference_answer": "Zcoder is built on React.js, Node.js, Express.js, MongoDB, and Socket.IO. It secures users via stateless JWT authentication, bcrypt password hashing, and email verification using Nodemailer with UUID-based tokens. It supports secure multi-language code execution and solution submission across 100+ coding problems, with modular RESTful APIs and Mongoose models for problems, contests, and profiles, and provides real-time user-to-user chat via bidirectional Socket.IO communication.",
            "expected_facts": [
                "Zcoder technology stack consists of React.js, Node.js, Express.js, MongoDB, and Socket.IO.",
                "Authentication uses stateless JWT authentication, bcrypt password hashing, and email verification using Nodemailer with UUID-based tokens.",
                "Code execution supports secure multi-language code execution and solution submission across 100+ coding problems.",
                "Backend features modular RESTful APIs and Mongoose schema models for problem management, contests, bookmarking, and profiles.",
                "Real-time user-to-user chat is built using Socket.IO event-driven bidirectional communication."
            ]
        },
        {
            "question_id": "Q09",
            "category": "Category C: Multi-part",
            "category_code": "C",
            "question": "Explain the Dolphin Student Opportunity Portal in terms of its frontend architecture, backend API, database, and core features.",
            "is_answerable": True,
            "relevant_chunk_ids": [14, 15],
            "reference_answer": "Dolphin is built with a responsive React.js frontend (using React Router and Axios with clean client-server separation) and a Node.js/Express.js backend with MongoDB. It features secure RESTful APIs with JWT authentication and bcrypt, aggregating 50+ hackathons, internships, workshops, and competitions with search, filtering, and personalized bookmarking management.",
            "expected_facts": [
                "Frontend is built with responsive React.js using React Router and Axios with clean client-server separation.",
                "Backend consists of secure RESTful APIs built with Node.js and Express.js.",
                "Authentication and security use JWT authentication and bcrypt password hashing.",
                "Database uses MongoDB for data models.",
                "Core functionality aggregates 50+ hackathons, internships, workshops, and coding competitions with search, filtering, and personalized bookmarking."
            ]
        },
        {
            "question_id": "Q10",
            "category": "Category D: Multi-section synthesis",
            "category_code": "D",
            "question": "Summarize my technical background, projects, and coding achievements across my resume.",
            "is_answerable": True,
            "relevant_chunk_ids": [11, 12, 13, 14, 15],
            "reference_answer": "The candidate is a Computer Science and Engineering (AI & ML) student at JNTUH (CGPA 8.88/10, Class of 2027) with skills in Python, Java, C++, OOP, design patterns, FastAPI, Node.js, React.js, Bedrock, ChromaDB, MySQL, and Snowflake. Key projects are Insurance AI Assistant (RAG with Titan Embeddings and Bedrock Nova Lite), Zcoder (coding platform with 100+ problems and Socket.IO chat), and Dolphin (opportunity portal aggregating 50+ events). Achievements include solving 600+ DSA problems (LeetCode 300+, GFG 300+) and a Certificate of Appreciation from Walmart Converge Hackathon.",
            "expected_facts": [
                "Education: B.Tech in Computer Science and Engineering (AI & ML), Class of 2027 at JNTUH with CGPA 8.88/10.",
                "Technical skills include Python, Java, C++, OOP, SOLID, design patterns, FastAPI, Node.js, React.js, MySQL, MongoDB, Snowflake, ChromaDB, Bedrock, and AWS S3.",
                "Insurance AI Assistant project features a RAG pipeline with PyMuPDF, Titan Embeddings V2, ChromaDB, Bedrock Nova 2 Lite, S3, MySQL, and Snowflake.",
                "Zcoder project features multi-language code execution for 100+ problems, JWT auth, Nodemailer verification, and Socket.IO chat.",
                "Dolphin project is an opportunity portal aggregating 50+ hackathons and internships with React, Node.js, Express, and MongoDB.",
                "Achievements include solving 600+ DSA problems across LeetCode (300+) and GeeksforGeeks (300+), and a Certificate of Appreciation from the Walmart Converge Hackathon."
            ]
        },
        {
            "question_id": "Q11",
            "category": "Category D: Multi-section synthesis",
            "category_code": "D",
            "question": "Identify the projects demonstrating backend development, AI/ML engineering, and real-time communication, and explain the technologies used in each.",
            "is_answerable": True,
            "relevant_chunk_ids": [11, 12, 13, 14],
            "reference_answer": "AI/ML engineering and backend development are demonstrated by the Insurance AI Assistant using Python, FastAPI, Amazon Bedrock Nova 2 Lite, Titan Text Embeddings V2, ChromaDB, and AWS S3. Real-time communication and backend development are demonstrated by Zcoder using Socket.IO for bidirectional chat along with Node.js, Express.js, MongoDB, and JWT authentication. Backend development is also demonstrated by Dolphin using Node.js, Express.js, and MongoDB.",
            "expected_facts": [
                "AI/ML engineering and backend are demonstrated by the Insurance AI Assistant using Python, FastAPI, Amazon Bedrock Nova 2 Lite, Titan Text Embeddings V2, ChromaDB, and AWS S3.",
                "Real-time communication and backend are demonstrated by Zcoder using Socket.IO for event-driven bidirectional chat along with Node.js, Express.js, MongoDB, and JWT.",
                "Backend development is also demonstrated by Dolphin using Node.js and Express.js RESTful APIs with JWT, bcrypt, and MongoDB."
            ]
        },
        {
            "question_id": "Q12",
            "category": "Category D: Multi-section synthesis",
            "category_code": "D",
            "question": "Detail the authentication and security mechanisms I have implemented across all projects on my resume.",
            "is_answerable": True,
            "relevant_chunk_ids": [11, 12, 13, 14],
            "reference_answer": "Across the projects, stateless JWT authentication is implemented for session and API security (in Zcoder and Dolphin), bcrypt is used for password hashing (in Zcoder and Dolphin), and Nodemailer with UUID-based tokens is used for email verification in Zcoder.",
            "expected_facts": [
                "Stateless JWT authentication is implemented across projects including Zcoder and Dolphin.",
                "bcrypt password hashing is used for secure user password protection in Zcoder and Dolphin.",
                "Email verification using Nodemailer with UUID-based tokens is implemented in Zcoder."
            ]
        },
        {
            "question_id": "Q13",
            "category": "Category E: Unanswerable (Abstention)",
            "category_code": "E",
            "question": "What exact percentage improvement in document retrieval accuracy did my Insurance AI Assistant achieve compared to standard keyword search?",
            "is_answerable": False,
            "relevant_chunk_ids": [],
            "reference_answer": "The document does not provide information regarding percentage improvement in retrieval accuracy or comparison against keyword search.",
            "expected_facts": [
                "The document does not provide information regarding percentage improvement in retrieval accuracy or comparison against keyword search."
            ]
        },
        {
            "question_id": "Q14",
            "category": "Category E: Unanswerable (Abstention)",
            "category_code": "E",
            "question": "What was the measured production end-to-end latency in milliseconds and AWS monthly infrastructure cost of my RAG pipeline?",
            "is_answerable": False,
            "relevant_chunk_ids": [],
            "reference_answer": "The document does not provide information regarding production latency in milliseconds or AWS monthly infrastructure costs for the RAG pipeline.",
            "expected_facts": [
                "The document does not provide information regarding production latency in milliseconds or AWS monthly infrastructure costs."
            ]
        },
        {
            "question_id": "Q15",
            "category": "Category E: Unanswerable (Abstention)",
            "category_code": "E",
            "question": "What was the exact F1 score and precision of the computer vision model in the Stampede Recognition project?",
            "is_answerable": False,
            "relevant_chunk_ids": [],
            "reference_answer": "The uploaded document does not mention a Stampede Recognition project or any computer vision model F1 score and precision.",
            "expected_facts": [
                "The uploaded document does not mention a Stampede Recognition project or any computer vision model F1 score and precision."
            ]
        }
    ]
}

target_path = Path("c:/Insurance_AI/evaluation_results/question_ground_truth.json")
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=2)

print(f"Saved ground truth dataset to {target_path} with {len(ground_truth['questions'])} questions.")
