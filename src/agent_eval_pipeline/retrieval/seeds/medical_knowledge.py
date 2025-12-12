"""
Medical knowledge base seed data.

This module contains the initial documents for the RAG system.
In production, this would come from a proper content pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_eval_pipeline.retrieval.document import Document

if TYPE_CHECKING:
    from agent_eval_pipeline.core import VectorStore


def get_medical_documents() -> list[Document]:
    """
    Get seed documents for the medical knowledge base.

    These documents cover thyroid function, which aligns with our
    golden test cases. In production, this would load from a database
    or content management system.
    """
    return [
        Document(
            id="doc_thyroid_101",
            title="Understanding Thyroid Function Tests",
            content="""TSH (Thyroid Stimulating Hormone) is the primary screening test for thyroid function.
It's produced by the pituitary gland and tells your thyroid to produce hormones.

Normal TSH range: 0.4-4.0 mIU/L

High TSH (>4.0):
- May indicate hypothyroidism (underactive thyroid)
- Pituitary is working harder to stimulate a sluggish thyroid
- Common symptoms: fatigue, weight gain, cold intolerance

Low TSH (<0.4):
- May indicate hyperthyroidism (overactive thyroid)
- Pituitary is backing off because thyroid is overproducing
- Common symptoms: weight loss, rapid heartbeat, anxiety

Important: TSH should be interpreted alongside Free T4 and Free T3.""",
            markers=["TSH", "Free T4", "Free T3"],
        ),
        Document(
            id="doc_tsh_interpretation",
            title="Interpreting TSH Results in Clinical Context",
            content="""TSH interpretation requires clinical context:

Subclinical hypothyroidism:
- TSH slightly elevated (4.5-10 mIU/L)
- Free T4 normal
- May or may not need treatment

Primary hypothyroidism:
- TSH elevated
- Free T4 low
- Usually requires treatment

TSH trends over time are often more informative than single values.
A rising TSH trend, even within normal range, may warrant attention.""",
            markers=["TSH"],
        ),
        Document(
            id="doc_t4_basics",
            title="Free T4: The Active Thyroid Hormone",
            content="""Free T4 (thyroxine) is the main hormone produced by the thyroid.

Normal range: 0.8-1.8 ng/dL

Free T4 is "free" because it's not bound to proteins and is available
for the body to use. It's converted to T3 in tissues.

Interpretation with TSH:
- Low T4 + High TSH = Primary hypothyroidism
- High T4 + Low TSH = Hyperthyroidism
- Low T4 + Low TSH = Central hypothyroidism (rare, pituitary issue)""",
            markers=["Free T4", "TSH"],
        ),
        Document(
            id="doc_hyperthyroid_patterns",
            title="Recognizing Hyperthyroidism Lab Patterns",
            content="""Hyperthyroidism occurs when the thyroid produces too much hormone.

Classic lab pattern:
- TSH: Low or suppressed (<0.4 mIU/L)
- Free T4: Elevated (>1.8 ng/dL)
- Free T3: Often elevated

Common causes:
- Graves' disease (autoimmune)
- Toxic nodular goiter
- Thyroiditis (temporary)

Symptoms to watch for:
- Unexplained weight loss
- Rapid or irregular heartbeat
- Nervousness, anxiety
- Heat intolerance
- Tremor

Requires medical evaluation and often treatment.""",
            markers=["TSH", "Free T4", "Free T3"],
        ),
        Document(
            id="doc_subclinical_thyroid",
            title="Subclinical Thyroid Disorders",
            content="""Subclinical thyroid disorders are when TSH is abnormal but T4/T3 are normal.

Subclinical hypothyroidism:
- TSH: 4.5-10 mIU/L
- Free T4: Normal
- May progress to overt hypothyroidism
- Treatment is controversial for mild cases

Borderline values (TSH 4.0-4.5):
- Often normal variation
- Worth monitoring over time
- Consider retesting in 6-12 weeks

Factors that affect TSH:
- Time of day (higher in morning)
- Stress and illness
- Medications (biotin can interfere)
- Age (TSH tends to increase with age)""",
            markers=["TSH"],
        ),
        Document(
            id="doc_vitamin_d",
            title="Vitamin D and Overall Health",
            content="""Vitamin D is essential for bone health, immune function, and mood.

Optimal levels: 30-50 ng/mL
Deficiency: <20 ng/mL
Insufficiency: 20-29 ng/mL

Symptoms of deficiency:
- Fatigue and tiredness
- Bone pain and weakness
- Muscle weakness
- Depression

Common causes of deficiency:
- Limited sun exposure
- Dark skin (reduces vitamin D synthesis)
- Age (skin becomes less efficient)
- Obesity (vitamin D gets stored in fat)

Treatment typically involves supplementation, with dosing based on
severity of deficiency. Retest after 2-3 months of supplementation.""",
            markers=["Vitamin D"],
        ),
    ]


def seed_vector_store(store: VectorStore) -> None:
    """
    Seed a vector store with medical documents.

    This function is agnostic to the store implementation - it works
    with PgVectorStore, InMemoryVectorStore, or any other implementation.

    Args:
        store: Any VectorStore implementation
    """
    docs = get_medical_documents()

    # Use batch insert if available, otherwise insert one by one
    if hasattr(store, "insert_documents_batch"):
        store.insert_documents_batch(docs)
    elif hasattr(store, "insert_document"):
        for doc in docs:
            store.insert_document(doc)
    else:
        raise TypeError(
            f"Store {type(store).__name__} does not support document insertion"
        )
