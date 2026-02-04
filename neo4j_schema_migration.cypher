// Neo4j Aura Free Schema Migration for Luxia KG Integration
// Execute these commands in Neo4j Browser or cypher-shell
// 1. Node Constraints (unique identifiers)
CREATE CONSTRAINT claim_id_unique IF NOT EXISTS
FOR (c:Claim)
REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity)
REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT relation_rid_unique IF NOT EXISTS
FOR (r:Relation)
REQUIRE r.rid IS UNIQUE;

CREATE CONSTRAINT source_url_unique IF NOT EXISTS
FOR (s:Source)
REQUIRE s.url IS UNIQUE;

// 2. Node Indexes (for fast lookups)
CREATE INDEX claim_text_index IF NOT EXISTS
FOR (c:Claim)
ON (c.text);

CREATE INDEX entity_name_index IF NOT EXISTS
FOR (e:Entity)
ON (e.name);

CREATE INDEX relation_predicate_index IF NOT EXISTS
FOR (r:Relation)
ON (r.predicate);

// 3. Source domain index
CREATE INDEX source_domain_index IF NOT EXISTS
FOR (s:Source)
ON (s.domain);

// 4. Relationship Indexes (for traversal performance)
CREATE INDEX subject_of_relationship IF NOT EXISTS
FOR ()-[s:SUBJECT_OF]-()
ON (s.confidence);

CREATE INDEX object_of_relationship IF NOT EXISTS
FOR ()-[o:OBJECT_OF]-()
ON (o.confidence);

CREATE INDEX supported_by_relationship IF NOT EXISTS
FOR ()-[sup:SUPPORTED_BY]-()
ON (sup.evidence_type);

// 5. Property Indexes (for query performance)
CREATE INDEX relation_confidence_index IF NOT EXISTS
FOR (r:Relation)
ON (r.confidence);

CREATE INDEX relation_updated_at_index IF NOT EXISTS
FOR (r:Relation)
ON (r.updated_at);
CREATE INDEX relation_confidence_index IF NOT EXISTS
FOR (r:Relation)
ON (r.confidence);

CREATE INDEX relation_updated_at_index IF NOT EXISTS
FOR (r:Relation)
ON (r.updated_at);

// Verification queries (run after migration):
// MATCH (c:Claim) RETURN count(c) as claims;
// MATCH (e:Entity) RETURN count(e) as entities;
// MATCH (r:Relation) RETURN count(r) as relations;
// MATCH (s:Source) RETURN count(s) as sources;
// CALL db.constraints() YIELD name, labelsOrTypes, properties, ownedIndex;
// CALL db.indexes() YIELD name, labelsOrTypes, properties, type;