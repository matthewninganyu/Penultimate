import type { AnnotationDocument, PersistentAnnotation } from "./annotationModel";

type DocumentState = PersistentAnnotation[];

function cloneAnnotations(annotations: PersistentAnnotation[]): DocumentState {
  return structuredClone(annotations);
}

function statesMatch(left: DocumentState, right: DocumentState) {
  return JSON.stringify(left) === JSON.stringify(right);
}

export class AnnotationHistory {
  private readonly past: DocumentState[] = [];
  private readonly future: DocumentState[] = [];
  private pending: DocumentState | null = null;

  get canUndo() {
    return this.past.length > 0;
  }

  get canRedo() {
    return this.future.length > 0;
  }

  get transactionActive() {
    return this.pending !== null;
  }

  begin(document: AnnotationDocument) {
    if (this.pending) return;
    this.pending = cloneAnnotations(document.annotations);
  }

  commit(document: AnnotationDocument) {
    if (!this.pending) return;
    const before = this.pending;
    this.pending = null;

    if (statesMatch(before, document.annotations)) return;
    this.past.push(before);
    this.future.length = 0;
  }

  cancel() {
    this.pending = null;
  }

  undo(document: AnnotationDocument) {
    this.pending = null;
    const previous = this.past.pop();
    if (!previous) return false;

    this.future.push(cloneAnnotations(document.annotations));
    document.annotations.splice(
      0,
      document.annotations.length,
      ...cloneAnnotations(previous),
    );
    return true;
  }

  redo(document: AnnotationDocument) {
    this.pending = null;
    const next = this.future.pop();
    if (!next) return false;

    this.past.push(cloneAnnotations(document.annotations));
    document.annotations.splice(
      0,
      document.annotations.length,
      ...cloneAnnotations(next),
    );
    return true;
  }

  reset() {
    this.past.length = 0;
    this.future.length = 0;
    this.pending = null;
  }
}
