// Neo4j Aura Free Schema Migration for Luxia KG Integration
// Execute in Neo4j Browser (Aura)
// 1) Unique constraints (stable identifiers)
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity)
REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT relation_rid_unique IF NOT EXISTS
FOR (r:Relation)
REQUIRE r.rid IS UNIQUE;

CREATE CONSTRAINT source_url_unique IF NOT EXISTS
FOR (s:Source)
REQUIRE s.url IS UNIQUE;

// OPTIONAL (only if you will create Claim nodes later)
CREATE CONSTRAINT claim_id_unique IF NOT EXISTS
FOR (c:Claim)
REQUIRE c.id IS UNIQUE;

// 2) Node indexes (fast lookups / filters)
CREATE INDEX entity_name_index IF NOT EXISTS
FOR (e:Entity)
ON (e.name);

CREATE INDEX relation_predicate_index IF NOT EXISTS
FOR (r:Relation)
ON (r.predicate);

CREATE INDEX source_domain_index IF NOT EXISTS
FOR (s:Source)
ON (s.domain);

// 3) Relation-node property indexes (useful for ORDER BY / filtering)
CREATE INDEX relation_confidence_index IF NOT EXISTS
FOR (r:Relation)
ON (r.confidence);

CREATE INDEX relation_updated_at_index IF NOT EXISTS
FOR (r:Relation)
ON (r.updated_at);

// OPTIONAL (only if you will store Claim.text later)
CREATE INDEX claim_text_index IF NOT EXISTS
FOR (c:Claim)
ON (c.text);

SHOW CONSTRAINTS;
SHOW INDEXES;

MATCH (e:Entity)
RETURN count(e) AS entities;
MATCH (r:Relation)
RETURN count(r) AS relations;
MATCH (s:Source)
RETURN count(s) AS sources;
MATCH (:Entity)-[:SUBJECT_OF]->(:Relation)-[:OBJECT_OF]->(:Entity)
RETURN count(*) AS triples;